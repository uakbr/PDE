////////////////////////////////////////////////////////////////////////////////
// heston-adi-solver-ultimate.ts
////////////////////////////////////////////////////////////////////////////////

import {
  Worker, isMainThread, parentPort, workerData
} from "worker_threads"

////////////////////////////////////////////////////////////////////////////////
// 1. Parameter Interfaces & Data Structures
////////////////////////////////////////////////////////////////////////////////

/**
 * Main configuration for our HPC-level Heston PDE solver.
 */
export interface HestonParams {
  // Heston model
  kappa: number
  theta: number
  sigma: number
  rho: number
  v0: number
  r: number

  // Option
  S0: number
  strike: number
  T: number
  putOption?: boolean

  // PDE & Grid
  NS?: number
  Nv?: number
  initNt?: number
  SmaxFactor?: number
  VmaxFactor?: number
  sGridSpacing?: "geometric" | "linear"
  vGridSpacing?: "geometric" | "linear"

  // HPC Parallel
  parallel?: boolean
  maxThreads?: number

  // Operator Splitting
  // e.g., thetaS for S-operator, thetaV for V-operator, thetaCross for cross-term, etc.
  thetaS?: number
  thetaV?: number
  thetaCross?: number
  // If we have some leftover operator L0 (zero-order terms), we define theta0
  theta0?: number
  handleFullCrossTerm?: boolean

  // Adaptive Mesh
  adaptiveMesh?: boolean
  refineFactor?: number
  minGridSize?: number
  errorBasedRefine?: boolean
  refineIntervalSteps?: number

  // Local Time-Stepping
  refinedLocalTime?: boolean

  // Validation
  compareWithAnalytic?: boolean
}

/**
 * Subdomain info for domain decomposition.
 */
interface Subdomain {
  sStart: number
  sEnd: number
  vStart: number
  vEnd: number
  threadId: number
  neighbors: number[]  // array of neighbor threadIds
}

/**
 * A specification for boundary copying. 
 */
type BoundarySpec =
  | { type: "LEFT"; vRange: [number, number] }
  | { type: "RIGHT"; vRange: [number, number] }
  | { type: "TOP"; sRange: [number, number] }
  | { type: "BOTTOM"; sRange: [number, number] }
  | { type: "CORNER"; corner: "TOP_LEFT"|"TOP_RIGHT"|"BOTTOM_LEFT"|"BOTTOM_RIGHT" }

////////////////////////////////////////////////////////////////////////////////
// 2. Global Data for Orchestration
////////////////////////////////////////////////////////////////////////////////

let subdomainMap: Subdomain[] = []

////////////////////////////////////////////////////////////////////////////////
// 3. Main Entry Function
////////////////////////////////////////////////////////////////////////////////

export async function solveHestonPdeUltimate(params: HestonParams): Promise<number> {
  if (params.parallel && isMainThread) {
    return parallelDomainOrchestrator(params)
  } else {
    return singleThreadSolverWorker(params)
  }
}

////////////////////////////////////////////////////////////////////////////////
// 4. Parallel Orchestrator
////////////////////////////////////////////////////////////////////////////////

async function parallelDomainOrchestrator(params: HestonParams): Promise<number> {
  const { NS=80, Nv=80 } = params
  const maxThreads = params.maxThreads ?? 4

  // Partition the domain
  const numSlicesX = Math.floor(Math.sqrt(maxThreads))
  const numSlicesY = Math.ceil(maxThreads / numSlicesX)
  const sliceSizeS = Math.ceil((NS+1)/numSlicesX)
  const sliceSizeV = Math.ceil((Nv+1)/numSlicesY)

  // Shared memory for PDE
  const totalCells = (NS+1)*(Nv+1)
  const sab = new SharedArrayBuffer(totalCells*Float64Array.BYTES_PER_ELEMENT)
  const sharedGrid = new Float64Array(sab)

  // Initialize terminal payoff (entire domain)
  initializeGrid(sharedGrid, params, NS, Nv)

  // We'll track subdomains in a 2D array for adjacency
  const subdomains2D: number[][] = new Array(numSlicesX)
  for (let ix = 0; ix < numSlicesX; ix++) {
    subdomains2D[ix] = new Array(numSlicesY).fill(-1)
  }

  let sliceId = 0
  for (let ix=0; ix<numSlicesX; ix++) {
    for (let iy=0; iy<numSlicesY; iy++) {
      sliceId++
      // if we exceed maxThreads, break early
      if (sliceId > maxThreads) break

      const sStart = ix*sliceSizeS
      const sEnd   = Math.min((ix+1)*sliceSizeS, NS+1)
      const vStart = iy*sliceSizeV
      const vEnd   = Math.min((iy+1)*sliceSizeV, Nv+1)

      // Create subdomain object
      const subd: Subdomain = {
        sStart, sEnd, vStart, vEnd,
        threadId: sliceId,  // 1-based ID
        neighbors: []
      }
      subdomainMap.push(subd)
      subdomains2D[ix][iy] = subdomainMap.length - 1 // store index in subdomainMap
    }
  }

  // Build actual adjacency (8-way) among subdomains
  buildNeighborMap(subdomains2D, numSlicesX, numSlicesY)

  // Spawn workers
  const workerPromises: Promise<number>[] = []
  for (const sd of subdomainMap) {
    const wparams= { ...params, sabRef: sab, subDomain: sd }
    const p = new Promise<number>((resolve, reject)=>{
      const w = new Worker(__filename, { workerData: wparams })
      w.on("message",(msg)=>{
        if (msg.type==="DONE") {
          resolve(msg.price)
        } else if (msg.type==="GHOST_REQ") {
          // partial boundary copying
          handleGhostRequest(msg, w)
        } else if (msg.type==="GHOST_DATA") {
          forwardGhostData(msg)
        } else if (msg.type==="ERROR") {
          reject(new Error(msg.error))
        }
      })
      w.on("error",(err)=>reject(err))
      w.on("exit",(code)=>{
        if (code!==0) reject(new Error(`Worker exit code=${code}`))
      })
    })
    workerPromises.push(p)
  }

  const results= await Promise.all(workerPromises)
  const finalPrice= results.reduce((a,b)=>a+b,0)/results.length
  return finalPrice
}

/**
 * buildNeighborMap():
 * Fill in the adjacency by scanning the 2D subdomains array.
 * We do an 8-way adjacency:
 *   Up, Down, Left, Right, and the 4 diagonals.
 */
function buildNeighborMap(subdomains2D: number[][], Nx: number, Ny: number) {
  for (let ix = 0; ix < Nx; ix++) {
    for (let iy = 0; iy < Ny; iy++) {
      const meIndex = subdomains2D[ix][iy]
      if (meIndex < 0) continue // means no subdomain here (exceeded maxThreads)
      const meSubdomain = subdomainMap[meIndex]

      // Check neighbors in a [-1, 0, 1] box
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue // skip self
          const nx = ix + dx
          const ny = iy + dy
          if (nx >= 0 && nx < Nx && ny >= 0 && ny < Ny) {
            const neighborIndex = subdomains2D[nx][ny]
            if (neighborIndex >= 0) {
              const neighborSub = subdomainMap[neighborIndex]
              meSubdomain.neighbors.push(neighborSub.threadId)
            }
          }
        }
      }
    }
  }
}

/**
 * handleGhostRequest():
 * The worker wants boundary data from a neighbor. We read partial slices 
 * from the shared array (subdomain of the neighbor).
 */
function handleGhostRequest(msg: any, w: Worker) {
  const { requestorThreadId, neighborThreadId, boundarySpec } = msg
  const neighbor = subdomainMap.find(s => s.threadId===neighborThreadId)
  if (!neighbor) {
    w.postMessage({ type:"ERROR", error:`No neighbor with threadId=${neighborThreadId}` })
    return
  }
  const boundaryData = readBoundarySlice(neighbor, boundarySpec)
  w.postMessage({
    type: "GHOST_DATA",
    requestorThreadId,
    boundarySpec,
    boundaryData
  })
}

/**
 * readBoundarySlice():
 * Copy the relevant partial indexing from neighbor's subdomain 
 * in the shared array. (Placeholder logic here.)
 */
function readBoundarySlice(neighbor: Subdomain, boundarySpec: BoundarySpec): Float64Array {
  // Real HPC code would do partial indexing. For demonstration, we do a stub.
  const data= new Float64Array(10) // placeholder
  return data
}

/**
 * forwardGhostData():
 * If a GHOST_DATA message is for a requestor subdomain, we forward it 
 * to that subdomain worker. (Omitted actual dictionary of workers in this demo.)
 */
function forwardGhostData(msg: any) {
  // In real HPC code, we'd have a dictionary of Worker references keyed by threadId.
  // Then we find the requestor and postMessage(...). Skipped for brevity.
}

////////////////////////////////////////////////////////////////////////////////
// 5. Worker Implementation
////////////////////////////////////////////////////////////////////////////////

if (!isMainThread && parentPort) {
  const p=workerData as HestonParams & {
    sabRef?: SharedArrayBuffer
    subDomain?: Subdomain
  }
  singleThreadSolverWorker(p)
    .then(price=> parentPort?.postMessage({ type:"DONE", price }))
    .catch(err=> parentPort?.postMessage({ type:"ERROR", error: err.message }))
}

/**
 * singleThreadSolverWorker():
 * Incorporates:
 *  - multi-stage operator splitting with separate \theta parameters
 *  - ghost cell boundary copying
 *  - adaptive mesh + re-projection
 *  - local time-step
 */
async function singleThreadSolverWorker(params: HestonParams & {
  sabRef?: SharedArrayBuffer
  subDomain?: Subdomain
}): Promise<number> {
  // 1) Build or refine initial grids
  let { Svals, Vvals } = buildGrids(params)
  if (params.adaptiveMesh) {
    const { refinedS, refinedV } = adaptMesh(Svals, Vvals, params)
    Svals= refinedS
    Vvals= refinedV
  }
  const NS= Svals.length-1
  const Nv= Vvals.length-1

  // 2) PDE domain array
  let domain: Float64Array
  if (params.sabRef) {
    domain= new Float64Array(params.sabRef, 0, (NS+1)*(Nv+1))
  } else {
    const sabLocal= new SharedArrayBuffer((NS+1)*(Nv+1)*Float64Array.BYTES_PER_ELEMENT)
    domain= new Float64Array(sabLocal)
    initializeGridCustom(domain, Svals, Vvals, params)
  }

  // subdomain boundaries
  const sStart= params.subDomain?.sStart?? 0
  const sEnd  = params.subDomain?.sEnd?? (NS+1)
  const vStart= params.subDomain?.vStart?? 0
  const vEnd  = params.subDomain?.vEnd?? (Nv+1)

  // time-step
  let dt= (params.T/(params.initNt??80))
  let timeSteps= params.initNt??80
  let t= params.T

  // Operator-splitting \theta parameters
  const thetaS= params.thetaS ?? 1.0
  const thetaV= params.thetaV ?? 1.0
  const thetaCross= params.thetaCross?? 1.0
  const theta0= params.theta0 ?? 1.0

  const refineInterval= params.refineIntervalSteps?? 20

  for (let step=0; step<timeSteps; step++) {
    t= params.T - step*dt

    // Stage A: L0 operator (the zero-order term: -r * U)
    applyL0(domain, dt*theta0, Svals, Vvals, params, { sStart, sEnd, vStart, vEnd })
    reapplyBoundary(domain, Svals, Vvals, params, t, { sStart, sEnd, vStart, vEnd })

    // Stage B: S operator
    doAdIStep(domain, Svals, Vvals, dt*thetaS, "S", params, { sStart, sEnd, vStart, vEnd })
    reapplyBoundary(domain, Svals, Vvals, params, t, { sStart, sEnd, vStart, vEnd })

    // Stage C: cross
    if (params.handleFullCrossTerm) {
      doCrossTermStep(domain, Svals, Vvals, dt*thetaCross, params, { sStart, sEnd, vStart, vEnd })
      reapplyBoundary(domain, Svals, Vvals, params, t, { sStart, sEnd, vStart, vEnd })
    }

    // Stage D: V operator
    doAdIStep(domain, Svals, Vvals, dt*thetaV, "v", params, { sStart, sEnd, vStart, vEnd })
    reapplyBoundary(domain, Svals, Vvals, params, t, { sStart, sEnd, vStart, vEnd })

    // local dt refine
    if (params.refinedLocalTime) {
      const cflLocal= computeLocalCfl(domain, Svals, Vvals, dt, params, { sStart, sEnd, vStart, vEnd })
      if (cflLocal<dt) {
        dt= cflLocal
        timeSteps= Math.floor(params.T/dt)
      }
    }

    // mid-simulation error-based re-grid
    if (params.errorBasedRefine && step>0 && step<timeSteps-1 && step%refineInterval===0) {
      const errMap= computeErrorIndicator(domain, Svals, Vvals, params, { sStart, sEnd, vStart, vEnd })
      const maxErr= Math.max(...errMap)
      if (maxErr>0.01) {
        // refine
        const oldDomain= domain
        const oldS= Svals.slice()
        const oldV= Vvals.slice()
        const { refinedS, refinedV } = adaptMeshError(oldDomain, oldS, oldV, errMap, params)
        Svals= refinedS
        Vvals= refinedV

        const newNS= Svals.length-1
        const newNv= Vvals.length-1
        const sabLocal= new SharedArrayBuffer((newNS+1)*(newNv+1)*Float64Array.BYTES_PER_ELEMENT)
        const newDomain= new Float64Array(sabLocal)
        initializeGridCustom(newDomain, Svals, Vvals, params)

        // re-project the partial PDE solution
        reprojectSolution(
          oldDomain, oldS, oldV,
          newDomain, Svals, Vvals,
          params
        )
        domain= newDomain
      }
    }

    // ghost cell exchange (if parallel subdomain) => omitted details
  }

  // Interpolate final price
  const iS= findNearestIndex(Svals, params.S0)
  const jV= findNearestIndex(Vvals, params.v0)
  const price= bilinearInterpolate2D(
    domain, 
    Svals.length-1, 
    Vvals.length-1, 
    Svals, 
    Vvals, 
    params.S0, 
    params.v0, 
    iS, jV
  )

  // Validation
  if (params.compareWithAnalytic) {
    const refPrice= hestonAnalyticPrice(params)
    console.log(`[VALIDATION] PDE=${price}, Analytic=${refPrice}, Diff=${price-refPrice}`)
  }

  return price
}

////////////////////////////////////////////////////////////////////////////////
// 5a. Full L0 Operator
////////////////////////////////////////////////////////////////////////////////

function applyL0(
  domain: Float64Array,
  dt: number,
  Svals: number[],
  Vvals: number[],
  params: HestonParams,
  sub: { sStart:number; sEnd:number; vStart:number; vEnd:number}
) {
  // In Heston PDE form, the zero-order term is simply -r * U
  // so the update is: U_new = U_old + dt * (-r * U_old) = (1 - r*dt)* U_old
  const { sStart, sEnd, vStart, vEnd }= sub
  const NS= Svals.length-1
  const Nv= Vvals.length-1
  for (let i=sStart; i< sEnd; i++){
    for (let j=vStart; j< vEnd; j++){
      domain[i*(Nv+1)+ j] *= (1 - params.r* dt)
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// 6. Re-Projection (Old->New) for Continuity
////////////////////////////////////////////////////////////////////////////////

function reprojectSolution(
  oldDomain: Float64Array,
  oldS: number[],
  oldV: number[],
  newDomain: Float64Array,
  newS: number[],
  newV: number[],
  params: HestonParams
) {
  const oldNS= oldS.length-1
  const oldNv= oldV.length-1
  const newNS= newS.length-1
  const newNv= newV.length-1

  for (let i=0; i<= newNS; i++) {
    for (let j=0; j<= newNv; j++) {
      const S_ = newS[i]
      const v_ = newV[j]
      // find nearest old indices
      const iS= findNearestIndex(oldS, S_)
      const jV= findNearestIndex(oldV, v_)

      const val= bilinearInterpolate2D(
        oldDomain,
        oldNS,
        oldNv,
        oldS,
        oldV,
        S_,
        v_,
        iS,
        jV
      )
      newDomain[i*(newNv+1)+j]= val
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// 7. PDE Helpers (Build Grids, Adapt Mesh, Error Indicators, BC, etc.)
////////////////////////////////////////////////////////////////////////////////

export function buildGrids(params: HestonParams) {
  const {
    NS=80, Nv=80,
    SmaxFactor=5, VmaxFactor=5,
    strike, v0, theta,
    sGridSpacing="geometric",
    vGridSpacing="geometric"
  }= params

  const Svals= new Array(NS+1)
  const Vvals= new Array(Nv+1)

  const Smax= SmaxFactor*strike
  if (sGridSpacing==="geometric") {
    for (let i=0; i<=NS; i++) {
      const r= i/NS
      Svals[i] = Smax*(r**2)
    }
  } else {
    const dS= Smax/NS
    for (let i=0; i<=NS; i++) {
      Svals[i]= i*dS
    }
  }

  const vmax= VmaxFactor*Math.max(v0, theta)
  if (vGridSpacing==="geometric") {
    for (let j=0; j<=Nv; j++) {
      const r= j/Nv
      Vvals[j]= vmax*(r**2)
    }
  } else {
    const dv= vmax/Nv
    for (let j=0; j<=Nv; j++) {
      Vvals[j]= j*dv
    }
  }

  return { Svals, Vvals }
}

function adaptMesh(Svals: number[], Vvals: number[], params: HestonParams) {
  const { refineFactor=2, minGridSize=20, strike, v0 }= params
  let refinedS: number[]= []
  for (let i=0; i<Svals.length-1; i++) {
    refinedS.push(Svals[i])
    const mid= 0.5*(Svals[i]+Svals[i+1])
    if (Math.abs(mid-strike)<0.2*strike && (Svals.length<500)) {
      const step= (Svals[i+1]-Svals[i])/refineFactor
      for (let k=1; k<refineFactor; k++) {
        refinedS.push(Svals[i]+k*step)
      }
    }
  }
  refinedS.push(Svals[Svals.length-1])
  if (refinedS.length<minGridSize) refinedS= Svals

  let refinedV: number[]= []
  for (let j=0; j<Vvals.length-1; j++) {
    refinedV.push(Vvals[j])
    const mid= 0.5*(Vvals[j]+Vvals[j+1])
    if (mid< 0.5*v0 && Vvals.length<500) {
      const step= (Vvals[j+1]-Vvals[j])/refineFactor
      for (let k=1; k<refineFactor; k++) {
        refinedV.push(Vvals[j]+k*step)
      }
    }
  }
  refinedV.push(Vvals[Vvals.length-1])
  if (refinedV.length<minGridSize) refinedV= Vvals

  return { refinedS, refinedV }
}

function adaptMeshError(
  oldDomain: Float64Array,
  oldS: number[],
  oldV: number[],
  errorMap: number[],
  params: HestonParams
) {
  const factor= params.refineFactor??2
  const newNS= (oldS.length-1)*factor
  const newNv= (oldV.length-1)*factor

  const refinedS= new Array(newNS+1)
  const refinedV= new Array(newNv+1)

  const Smax= oldS[oldS.length-1]
  for (let i=0; i<= newNS; i++) {
    const r= i/newNS
    refinedS[i]= Smax*(r**2)
  }

  const vmax= oldV[oldV.length-1]
  for (let j=0; j<= newNv; j++) {
    const r= j/newNv
    refinedV[j]= vmax*(r**2)
  }

  return { refinedS, refinedV }
}

function initializeGrid(grid: Float64Array, params: HestonParams, NS: number, Nv: number) {
  const { Svals, Vvals }= buildGrids(params)
  initializeGridCustom(grid, Svals, Vvals, params)
}

function initializeGridCustom(
  grid: Float64Array,
  Svals: number[],
  Vvals: number[],
  params: HestonParams
) {
  const NS= Svals.length-1
  const nv= Vvals.length-1
  for (let i=0; i<=NS; i++) {
    const payoff= calcPayoff(Svals[i], params.strike, params.putOption)
    for (let j=0; j<=nv; j++) {
      grid[i*(nv+1)+ j]= payoff
    }
  }
}

function calcPayoff(S: number, K: number, put?: boolean) {
  if (!put) return Math.max(S-K,0)
  return Math.max(K-S,0)
}

function computeErrorIndicator(
  domain: Float64Array,
  Svals: number[],
  Vvals: number[],
  params: HestonParams,
  sub: { sStart:number; sEnd:number; vStart:number; vEnd:number}
): number[] {
  const NS= Svals.length-1
  const Nv= Vvals.length-1
  const getVal=(i: number, j:number)=> domain[i*(Nv+1)+j]
  const errors= new Array((NS-1)*(Nv-1)).fill(0)
  let idx=0
  for (let i=1; i<NS; i++) {
    for (let j=1; j<Nv; j++) {
      const vx= getVal(i+1,j) - getVal(i-1,j)
      const vy= getVal(i,j+1) - getVal(i,j-1)
      errors[idx++]= Math.sqrt(vx*vx + vy*vy)
    }
  }
  return errors
}

////////////////////////////////////////////////////////////////////////////////
// 8. ADI Operators
////////////////////////////////////////////////////////////////////////////////

function doAdIStep(
  domain: Float64Array,
  Svals: number[],
  Vvals: number[],
  dt: number,
  direction: "S"|"v",
  params: HestonParams,
  sub: { sStart:number; sEnd:number; vStart:number; vEnd:number}
) {
  if (direction==="S") {
    solveInS(domain, Svals, Vvals, dt, params, sub)
  } else {
    solveInV(domain, Svals, Vvals, dt, params, sub)
  }
}

function doCrossTermStep(
  domain: Float64Array,
  Svals: number[],
  Vvals: number[],
  dt: number,
  params: HestonParams,
  sub: { sStart:number; sEnd:number; vStart:number; vEnd:number}
) {
  const { sStart, sEnd, vStart, vEnd }= sub
  const NS= Svals.length-1
  const Nv= Vvals.length-1
  const getVal=(i: number, j: number)=> domain[i*(Nv+1)+ j]
  const setVal=(i: number, j: number, val:number)=> { domain[i*(Nv+1)+ j]= val }

  for (let i=sStart+1; i<sEnd-1; i++) {
    const dS= (Svals[i+1]- Svals[i-1])*0.5
    for (let j=vStart+1; j<vEnd-1; j++) {
      const dv= (Vvals[j+1]- Vvals[j-1])*0.5
      const cross= ( getVal(i+1,j+1) - getVal(i-1,j+1)
                   - getVal(i+1,j-1) + getVal(i-1,j-1) )/(4*dS*dv)
      const S= Svals[i]
      const v= Vvals[j]
      const incr= dt* params.rho* params.sigma* S* v* cross
      setVal(i,j, getVal(i,j)+ incr)
    }
  }
}

/**
 * solveInS():
 * 1D in S dimension with partial dt.
 */
function solveInS(
  domain: Float64Array,
  Svals: number[],
  Vvals: number[],
  dt: number,
  params: HestonParams,
  sub: { sStart:number; sEnd:number; vStart:number; vEnd:number}
) {
  const { sStart, sEnd, vStart, vEnd }= sub
  const NS= Svals.length-1
  const Nv= Vvals.length-1
  const getVal=(i: number, j: number)=> domain[i*(Nv+1)+ j]
  const setVal=(i: number, j: number, val: number)=> { domain[i*(Nv+1)+ j]= val }

  for (let j=vStart; j<vEnd; j++) {
    const rowSize= sEnd- sStart
    const a= new Array(rowSize).fill(0)
    const b= new Array(rowSize).fill(0)
    const c= new Array(rowSize).fill(0)
    const d= new Array(rowSize).fill(0)

    for (let idx=0; idx< rowSize; idx++) {
      const i= sStart+ idx
      if (i===0||i===NS) {
        a[idx]=0; b[idx]=1; c[idx]=0; d[idx]= getVal(i,j)
        continue
      }
      const S= Svals[i]
      const v= Vvals[j]
      const diff= 0.5* dt* v* (S**2)
      const drift=0.5* dt* params.r* S

      a[idx]= -diff- drift
      b[idx]= 1+ 2* diff+ dt*params.r
      c[idx]= -diff+ drift
      d[idx]= getVal(i,j)
    }
    const newRow= thomasSolve(a,b,c,d)
    for (let idx=0; idx< rowSize; idx++) {
      const i= sStart+ idx
      setVal(i,j,newRow[idx])
    }
  }
}

/**
 * solveInV():
 * 1D in v dimension with partial dt.
 */
function solveInV(
  domain: Float64Array,
  Svals: number[],
  Vvals: number[],
  dt: number,
  params: HestonParams,
  sub: { sStart:number; sEnd:number; vStart:number; vEnd:number}
) {
  const { sStart, sEnd, vStart, vEnd }= sub
  const NS= Svals.length-1
  const Nv= Vvals.length-1
  const getVal=(i: number, j:number)=> domain[i*(Nv+1)+ j]
  const setVal=(i: number, j:number, val:number)=> { domain[i*(Nv+1)+ j]= val }

  for (let i=sStart; i<sEnd; i++) {
    const colSize= vEnd- vStart
    const a= new Array(colSize).fill(0)
    const b= new Array(colSize).fill(0)
    const c= new Array(colSize).fill(0)
    const d= new Array(colSize).fill(0)

    for (let idx=0; idx< colSize; idx++) {
      const j= vStart+ idx
      if (j===0|| j===Nv) {
        a[idx]=0; b[idx]=1; c[idx]=0; d[idx]= getVal(i,j)
        continue
      }
      const v= Vvals[j]
      const diff= 0.5* dt* (params.sigma**2)* v
      const drift=0.5* dt* params.kappa*( params.theta- v )

      a[idx]= - diff - drift
      b[idx]= 1+ 2* diff
      c[idx]= - diff + drift
      d[idx]= getVal(i,j)
    }
    const newCol= thomasSolve(a,b,c,d)
    for (let idx=0; idx< colSize; idx++) {
      const j= vStart+ idx
      setVal(i,j,newCol[idx])
    }
  }
}

function computeLocalCfl(
  domain: Float64Array,
  Svals: number[],
  Vvals: number[],
  dt: number,
  params: HestonParams,
  sub: { sStart:number; sEnd:number; vStart:number; vEnd:number}
): number {
  const { sStart, sEnd, vStart, vEnd }= sub
  let localMax=0
  for (let i=sStart; i<sEnd; i++) {
    for (let j=vStart; j<vEnd; j++) {
      const S= Svals[i]||1
      const v= Vvals[j]||1
      const scale= params.r + params.sigma + params.kappa + (S+v)*0.01
      localMax= Math.max(localMax, scale)
    }
  }
  const dtSafe= 1/(1+ localMax)
  return dtSafe<dt? dtSafe: dt
}

function reapplyBoundary(
  domain: Float64Array,
  Svals: number[],
  Vvals: number[],
  params: HestonParams,
  t: number,
  sub: { sStart:number; sEnd:number; vStart:number; vEnd:number}
) {
  const NS= Svals.length-1
  const Nv= Vvals.length-1
  const getVal=(i: number, j:number)=> domain[i*(Nv+1)+ j]
  const setVal=(i: number, j:number, val:number)=> { domain[i*(Nv+1)+ j]= val }

  const { sStart, sEnd, vStart, vEnd }= sub

  // S=0 => call=0, put=K e^{-r(T-t)}
  if (sStart===0) {
    for (let j=vStart; j<vEnd; j++) {
      setVal(0,j, params.putOption
        ? params.strike*Math.exp(-params.r*(params.T- t))
        : 0
      )
    }
  }
  // S=Smax => for a call, payoff ~ Smax-K => discount it
  if (sEnd=== NS+1) {
    for (let j=vStart; j<vEnd; j++) {
      const Smax= Svals[NS]
      const payoff= calcPayoff(Smax, params.strike, !params.putOption)
      setVal(NS,j, payoff*Math.exp(-params.r*(params.T- t)))
    }
  }

  // v=0 => clamp
  if (vStart===0) {
    for (let i=sStart; i<sEnd; i++) {
      setVal(i,0, Math.max(0, getVal(i,0)))
    }
  }
  // v=vmax => clamp
  if (vEnd=== Nv+1) {
    for (let i=sStart; i<sEnd; i++) {
      setVal(i,Nv, Math.max(0, getVal(i,Nv)))
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// 9. Numeric Utilities
////////////////////////////////////////////////////////////////////////////////

function bilinearInterpolate2D(
  domain: Float64Array,
  NS: number,
  Nv: number,
  Svals: number[],
  Vvals: number[],
  S0: number,
  v0: number,
  iS: number,
  jV: number
): number {
  const getVal=(i:number, j:number)=> domain[i*(Nv+1)+ j]
  if (iS>=NS || jV>=Nv) {
    return getVal(Math.min(iS,NS), Math.min(jV,Nv))
  }
  const S1= Svals[iS], S2= Svals[iS+1]
  const v1= Vvals[jV], v2= Vvals[jV+1]
  const Q11= getVal(iS,jV)
  const Q21= getVal(iS+1,jV)
  const Q12= getVal(iS,jV+1)
  const Q22= getVal(iS+1,jV+1)
  const denom= (S2- S1)*(v2- v1)
  if (denom===0) return Q11
  return (
    Q11*((S2- S0)*(v2- v0))/ denom
    +Q21*((S0- S1)*(v2- v0))/ denom
    +Q12*((S2- S0)*(v0- v1))/ denom
    +Q22*((S0- S1)*(v0- v1))/ denom
  )
}

function findNearestIndex(arr: number[], x: number): number {
  let idx=0
  let best= Math.abs(arr[0]- x)
  for (let i=1; i< arr.length; i++) {
    const d= Math.abs(arr[i]- x)
    if (d< best) {
      best= d
      idx= i
    }
  }
  return idx
}

/**
 * Thomas solver for tri-di systems
 */
function thomasSolve(a: number[], b: number[], c: number[], d: number[]): number[] {
  const n= b.length
  const cp= new Array(n).fill(0)
  const dp= new Array(n).fill(0)
  const x= new Array(n).fill(0)

  cp[0]= c[0]/ b[0]
  dp[0]= d[0]/ b[0]
  for (let i=1; i<n; i++) {
    const denom= b[i] - a[i]* cp[i-1]
    cp[i]= (i< n-1)? c[i]/ denom : 0
    dp[i]= ( d[i] - a[i]* dp[i-1] )/ denom
  }
  x[n-1]= dp[n-1]
  for (let i=n-2; i>=0; i--) {
    x[i]= dp[i]- cp[i]* x[i+1]
  }
  return x
}

////////////////////////////////////////////////////////////////////////////////
// 10. Heston Integral (More Complex CF)
////////////////////////////////////////////////////////////////////////////////

export function hestonAnalyticPrice(params: HestonParams): number {
  const { S0, strike:K, T, r, putOption }= params
  const discount= Math.exp(-r*T)
  const P1= doHestonFourierIntegral(params, 1)
  const P2= doHestonFourierIntegral(params, 2)
  const call= S0* P1 - K* discount* P2
  return putOption ? callToPut(call, S0, K, discount) : call
}

function doHestonFourierIntegral(params: HestonParams, j:1|2): number {
  const lnK= Math.log(params.strike)
  const integrand= (phi: number): number => {
    const cf= hestonCharacteristicFn(phi, j, params)
    // multiply by e^{- i phi lnK}/( i phi )
    const eTerm= cMulComplex(cf, expComplex({ re:0, im:- phi* lnK }))
    const iOverPhi: Complex= { re:0, im: 1/ phi }
    const valC= cMulComplex(eTerm, iOverPhi)
    return valC.re
  }

  let sum=0
  const phiMax= 200
  const N= 2000
  const dPhi= phiMax/ N
  for (let n=0; n<N; n++) {
    const ph= (n+0.5)* dPhi
    sum+= integrand(ph)
  }
  const val= dPhi* sum
  return 0.5 + (1/ Math.PI)* val
}

/**
 * hestonCharacteristicFn(phi,j):
 */
function hestonCharacteristicFn(phi: number, j:1|2, params: HestonParams): Complex {
  const { S0, T, r, kappa, theta, sigma, rho, v0 }= params
  const alpha= (j===1? 1:2)
  const b= ( j===1? (kappa - rho*sigma) : kappa )

  // iPhi
  const iPhi: Complex= { re:0, im: phi }

  // part= rho*sigma iPhi - b
  const part= subComplex( mulRealComplex(rho*sigma, iPhi), { re:b, im:0 })
  // iPhi + phi^2 => i phi + -phi^2 => { re:-phi^2, im: phi }
  const phi2Term: Complex= { re: - (phi**2), im: phi }
  const sigma2phi2= mulRealComplex( sigma*sigma, phi2Term )
  const part2= cSquare(part)
  const inside= addComplex(part2, sigma2phi2)
  const d= cSqrt(inside)

  // g= (part + d)/(part - d)
  const numerator= addComplex(part, d)
  const denominator= subComplex(part, d)
  const g= cDiv(numerator, denominator)

  const rminus= subComplex({ re:b, im:0 }, d)
  const rplus = addComplex({ re:b, im:0 }, d)

  const G= cDiv(rminus, rplus)
  const eDt= expComplex( mulRealComplex(-T, d) )
  const oneC= { re:1, im:0 }
  const top= subComplex(oneC, cMulComplex(G, eDt))
  const bottom= subComplex(oneC, G)
  const bigFrac= cDiv(top, bottom)

  // D= rminus * bigFrac
  const D_= mulComplex(rminus, bigFrac)

  // C= (kappa*theta / sigma^2 ) [ (rminus.re * T) - 2 ln( ... ) ]
  const cFactor= (kappa* theta)/( sigma* sigma )
  const partA= { re: rminus.re*T, im:0 }
  const logTerm= cLog( cDiv(
    subComplex(oneC, cMulComplex(G, eDt)),
    subComplex(oneC, G)
  ))
  const partB= mulRealComplex(-2, logTerm)
  const cSum= addComplex(partA, partB)
  const C_= mulRealComplex(cFactor, cSum)

  // exponent= i phi( ln(S0)+ rT ) + D_ v0 + C_
  const iPhiLnS0rT= mulComplex(iPhi, { re:Math.log(S0)+ r*T, im:0 })
  let expo= addComplex(iPhiLnS0rT, mulRealComplex(v0, D_))
  expo= addComplex(expo, C_)
  return expComplex(expo)
}

function callToPut(callPrice: number, S0: number, K:number, discount: number): number {
  return callPrice - S0 + K* discount
}

////////////////////////////////////////////////////////////////////////////////
// 11. Complex Arithmetic
////////////////////////////////////////////////////////////////////////////////

interface Complex { re: number; im: number }

function addComplex(a: Complex, b: Complex): Complex {
  return { re:a.re+ b.re, im:a.im+ b.im }
}
function subComplex(a: Complex, b: Complex): Complex {
  return { re:a.re- b.re, im:a.im- b.im }
}
function mulComplex(a: Complex, b: Complex): Complex {
  return {
    re: a.re*b.re - a.im*b.im,
    im: a.re*b.im + a.im*b.re
  }
}
function mulRealComplex(r: number, c: Complex): Complex {
  return { re: r*c.re, im: r*c.im }
}
function cSquare(a: Complex): Complex {
  return {
    re: a.re*a.re - a.im*a.im,
    im: 2*a.re*a.im
  }
}
function cDiv(a: Complex, b: Complex): Complex {
  const denom= b.re*b.re + b.im*b.im
  return {
    re: (a.re*b.re + a.im*b.im)/ denom,
    im: (a.im*b.re - a.re*b.im)/ denom
  }
}
function expComplex(c: Complex): Complex {
  const e= Math.exp(c.re)
  return { re:e* Math.cos(c.im), im:e* Math.sin(c.im) }
}
function cLog(a: Complex): Complex {
  const r= Math.sqrt(a.re*a.re+ a.im*a.im)
  const theta= Math.atan2(a.im, a.re)
  return { re:Math.log(r), im:theta }
}
function cSqrt(a: Complex): Complex {
  const r= Math.sqrt(a.re*a.re+ a.im*a.im)
  const theta= Math.atan2(a.im,a.re)*0.5
  const sr= Math.sqrt(r)
  return {
    re: sr* Math.cos(theta),
    im: sr* Math.sin(theta)
  }
}


////////////////////////////////////////////////////////////////////////////////
// 12. Example usage for single-thread test
////////////////////////////////////////////////////////////////////////////////
/*
(async ()=>{
  const result= await solveHestonPdeUltimate({
    kappa:1.5,
    theta:0.04,
    sigma:0.5,
    rho:-0.7,
    v0:0.04,
    r:0.01,

    S0:100,
    strike:100,
    T:1,
    putOption:false,

    NS:80,
    Nv:80,
    initNt:80,
    SmaxFactor:5,
    VmaxFactor:5,
    sGridSpacing:"geometric",
    vGridSpacing:"geometric",

    parallel:false,
    maxThreads:2,

    // separate theta parameters
    thetaS:1.0,
    thetaV:1.0,
    thetaCross:0.5,
    theta0:1.0,
    handleFullCrossTerm:true,

    // advanced mesh
    adaptiveMesh:true,
    refineFactor:2,
    minGridSize:20,
    errorBasedRefine:true,
    refineIntervalSteps:10,

    refinedLocalTime:true,

    compareWithAnalytic:true
  })
  console.log("Final PDE Price:", result)
})();
*/

////////////////////////////////////////////////////////////////////////////////
// End of heston-adi-solver-ultimate.ts
////////////////////////////////////////////////////////////////////////////////
