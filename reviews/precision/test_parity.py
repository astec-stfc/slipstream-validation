"""Numerical parity of Slipstream's backends: each path against eager float64 on the CPU, on the same lattice and particles.

A change in Slipstream's torch maps does not reach its Warp or double-single copies by itself, so every path is run here.  The Warp
ring kernel also runs on Warp's CPU backend, which puts it in reach of a runner without a GPU.

A path is a device (cpu, cuda, rocm), a precision (float64, float32, double-single) and a kernel (eager, torch.compile, Warp).
Compiled and Warp kernels may fuse a multiply and an add, or order a sum differently; the limits say how much that may change a result.
An accelerator is used when torch sees one: ROCm presents itself as ``torch.cuda`` and is labelled by ``torch.version.hip``.
Differences are the largest absolute difference of a coordinate over the particles, divided by its largest reference value.
"""
import functools
import inspect
import os
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("slipstream")

from slipstream.tracking.native import elements, ring_warp  # noqa: E402
from slipstream.tracking.native.collective import csr_steady_state_delta_kick_fast  # noqa: E402
from slipstream.tracking.native.lattice_import import ElementSpec as E  # noqa: E402
from slipstream.tracking.native.ring import RingMap  # noqa: E402
from slipstream.tracking.native.space_charge import SpaceCharge3D  # noqa: E402

pytestmark = pytest.mark.slipstream
F64, F32 = torch.float64, torch.float32
ACCEL = ("rocm" if getattr(torch.version, "hip", None) else "cuda") if torch.cuda.is_available() else None
on_accelerator = [pytest.mark.tier2, pytest.mark.skipif(ACCEL is None, reason="no GPU that torch can use")]
if ACCEL:
    torch.cuda.set_per_process_memory_fraction(0.5)   # the tests need megabytes; never spill the card into host memory
if os.name == "nt":
    os.environ.setdefault("CXX", "clang-cl")          # torch.compile on Windows (AGENTS.md)


# The marks of #62 and #59 follow the Slipstream under test, because the nightly runs two branches that need not have the same fixes.
WARP_TAU = ("astec-stfc/slipstream#62: ring_warp.py still forms the change of tau as arc / beta - ds / beta0, the difference of two "
            "nearly equal lengths that Slipstream #59 removed from the torch maps (elements._delay).  float32 on the Warp kernel loses tau.")
WARP_HAS_62 = "arc * inv_beta - ds * inv_beta0" in inspect.getsource(ring_warp)

# float32 tau of the torch maps: fixed by astec-stfc/slipstream#59, which a branch has once it has `elements._delay`
TAU_BEFORE_59 = pytest.mark.xfail(not hasattr(elements, "_delay"), strict=True, raises=AssertionError,
                                  reason="astec-stfc/slipstream#59: this Slipstream does not have the float32 tau fix yet")


def accelerator(test):
    """Skip without a GPU, and mark tier2."""
    for mark in on_accelerator:
        test = mark(test)
    return test


CELL = [E("QF", "quad", 0.2, k1=1.5), E("D1", "drift", 0.4), E("B", "bend", 1.0, angle=0.2, e1=0.1, e2=0.1),
        E("S", "sext", 0.1, k2=10.0), E("D2", "drift", 0.4), E("QD", "quad", 0.2, k1=-1.35), E("D3", "drift", 0.4)] * 2
TURNS = 20
COORDS = ("x", "px", "y", "py", "tau", "delta")
NOT_TAU = ("x", "px", "y", "py", "delta")

# Limits of the transverse coordinates and delta.  float64 and float32 are targets of about 100 and 30 times the unit round-off after
# 20 turns of this lattice; the double-single limits are Slipstream's own self-check (full) and a ratchet (hybrid, measured 9e-8).
LIMIT = {"f64": 1e-12, "f32": 1.5e-5, "ds-full": 1e-11, "ds-hybrid": 2e-7}


def particles():
    g = torch.Generator().manual_seed(1)
    scale = torch.tensor([1e-3, 1e-4, 5e-4, 5e-5, 0.0, 1e-3], dtype=F64)   # tau starts at zero: its error is that of its change
    return torch.randn(256, 6, dtype=F64, generator=g) * scale


def need_compiler():
    if shutil.which(os.environ.get("CXX", "g++")) is None:
        pytest.skip("torch.compile on the CPU needs a C++ compiler")


@functools.lru_cache(maxsize=None)
def track(device, precision, kernel):
    """State after TURNS turns, float64 on the CPU.  ``kernel``: eager, or default (what ``RingMap.track`` picks)."""
    ds = precision.startswith("ds-")
    ring = RingMap(CELL, 6e8, device=device, dtype=F64 if precision == "f64" else F32, n_kicks=2, n_kicks_bend=4, integration_order=4,
                   compile=kernel == "compiled" or (device == "cuda" and kernel == "default"),
                   double_single=precision.removeprefix("ds-") if ds else False)
    u = particles().to(device, F64 if precision == "f64" or ds else F32)
    with torch.no_grad():
        if ds or kernel != "eager":
            u = ring.track(u, TURNS, aperture=(1.0, 1.0)).state
        else:
            for _ in range(TURNS):
                u = ring.turn(u, eager=True)
    return u.double().cpu()


def differences(out, ref):
    return dict(zip(COORDS, ((out - ref).abs().amax(0) / ref.abs().amax(0).clamp_min(1e-300)).tolist()))


def check(record_property, device, precision, kernel, coords):
    found = differences(track(device, precision, kernel), track("cpu", "f64", "eager"))
    record_property("relative_difference", found)
    for c in coords:
        assert found[c] < LIMIT[precision], f"{c}: {found[c]:.2e} on {ACCEL if device == 'cuda' else 'cpu'}"


@pytest.mark.parametrize("precision,kernel,coords", [("f64", "compiled", COORDS), ("f32", "eager", NOT_TAU)], ids=["f64-compiled", "f32-eager"])
def test_cpu_ring(record_property, precision, kernel, coords):
    if kernel == "compiled":
        need_compiler()
    check(record_property, "cpu", precision, kernel, coords)


@TAU_BEFORE_59
def test_cpu_float32_tau(record_property):
    check(record_property, "cpu", "f32", "eager", ("tau",))


@pytest.mark.parametrize("precision", ["ds-full", "ds-hybrid"])
def test_cpu_double_single_ring(record_property, precision):
    check(record_property, "cpu", precision, "eager", ("x", "px", "y", "py", "delta"))   # tau is documented as not tracked
    assert torch.equal(track("cpu", precision, "eager")[:, 4], particles()[:, 4])


@accelerator
@pytest.mark.parametrize("precision,kernel", [("f64", "eager"), ("f64", "default"), ("f32", "eager"), ("f32", "default")])
def test_accelerator_ring(record_property, precision, kernel):
    check(record_property, "cuda", precision, kernel, NOT_TAU)


@accelerator
@pytest.mark.parametrize("precision,kernel", [("f64", "eager"), ("f64", "default"), pytest.param("f32", "eager", marks=TAU_BEFORE_59)])
def test_accelerator_tau(record_property, precision, kernel):
    check(record_property, "cuda", precision, kernel, ("tau",))


@accelerator
@pytest.mark.slow
@pytest.mark.parametrize("precision", ["ds-full", "ds-hybrid"])
def test_accelerator_double_single_ring(record_property, precision):
    need_compiler()
    check(record_property, "cuda", precision, "default", ("x", "px", "y", "py", "delta"))


def warp_ring_active():
    return ACCEL is not None and RingMap(CELL, 6e8, device="cuda", dtype=F32)._warp_params is not None


@accelerator
@pytest.mark.xfail(WARP_HAS_62 and warp_ring_active(), strict=True, raises=AssertionError, reason=WARP_TAU)
def test_accelerator_float32_tau_of_the_default_kernel(record_property):
    check(record_property, "cuda", "f32", "default", ("tau",))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "astec-stfc/slipstream#65: in double-single, the kick of a corrector given in its ElementSpec is dropped without an error.  A fix that "
    "refuses the lattice instead will fail this mark with NotImplementedError."))
@pytest.mark.parametrize("precision", ["ds-full", "ds-hybrid"])
def test_double_single_applies_a_corrector_kick(precision):
    """A path that does not implement an element must say so, not skip it."""
    spec, state = [E("K", "hkick", 0.2, kick_x=1e-3)], particles()
    ref = RingMap(spec, 6e8, device="cpu", dtype=F64).turn(state)
    ring = RingMap(spec, 6e8, device="cpu", double_single=precision.removeprefix("ds-"))
    assert differences(ring.track(state, 1, aperture=(1.0, 1.0)).state, ref)["px"] < LIMIT[precision]


# ------------------------------------------------------------------------------------------------ the Warp ring kernel, on the CPU
@functools.lru_cache(maxsize=None)
def warp_on_cpu(precision):
    """TURNS turns of the whole-turn Warp kernel on CPU tensors (``RingMap`` itself selects it on CUDA only)."""
    wp = pytest.importorskip("warp")
    from slipstream.tracking.native import ring_warp, warp_kernels

    dtype = F64 if precision == "f64" else F32
    ring = RingMap(CELL, 6e8, device="cpu", dtype=dtype, n_kicks=2, n_kicks_bend=4, integration_order=4)
    packed = ring_warp.pack(ring.params, torch.device("cpu"), dtype)
    assert packed is not None
    warp_kernels._ensure_init()
    u = particles().to(dtype)
    with pytest.MonkeyPatch.context() as patch, wp.ScopedDevice("cpu"):
        patch.setattr(warp_kernels, "_launch_stream", lambda tensor: None)   # a CUDA stream has no meaning on the CPU
        for _ in range(TURNS):
            u = ring_warp.turn(u, packed, ring.energy_eV, ring.mc2_eV)
    return u.double()


@pytest.mark.parametrize("precision,coords", [("f64", COORDS), ("f32", ("x", "px", "y", "py", "delta"))], ids=["f64", "f32"])
def test_warp_ring_kernel_on_cpu(record_property, precision, coords):
    found = differences(warp_on_cpu(precision), track("cpu", "f64", "eager"))
    record_property("relative_difference", found)
    for c in coords:
        assert found[c] < LIMIT[precision], f"{c}: {found[c]:.2e}"


@pytest.mark.xfail(WARP_HAS_62, strict=True, raises=AssertionError, reason=WARP_TAU)
def test_warp_ring_kernel_float32_tau_on_cpu():
    assert differences(warp_on_cpu("f32"), track("cpu", "f64", "eager"))["tau"] < LIMIT["f32"]


# ------------------------------------------------------------------------------------------------ collective kernels
def space_charge(device, dtype):
    g = torch.Generator().manual_seed(2)
    x, y, z = (torch.randn(20_000, dtype=F64, generator=g) * s for s in (1e-3, 1e-3, 3e-4))
    n, h = 32, (4e-4, 4e-4, 1.2e-4)
    solver = SpaceCharge3D(n, n, n, *h, *(-(n - 1) / 2 * hi for hi in h), device=device, dtype=dtype)
    p = [v.to(device, dtype) for v in (x, y, z, torch.full_like(x, 1e-10 / x.numel()))]
    E_grid = solver.solve(solver.deposit(*p))
    return torch.stack([solver.gather(E_grid[i], *p[:3]) for i in range(3)], -1).double().cpu()


def csr(device, dtype):
    g = torch.Generator().manual_seed(2)
    tau = torch.randn(20_000, dtype=F64, generator=g) * 2e-5
    energy = torch.tensor(190e6, dtype=dtype, device=device)
    return csr_steady_state_delta_kick_fast(tau.to(device, dtype), 100e-12 / tau.numel(), 4.0, energy, 0.2, n_bins=128).double().cpu()[:, None]


@pytest.mark.parametrize("kernel", [space_charge, csr])
@pytest.mark.parametrize("device,dtype,limit", [
    pytest.param("cpu", F32, 1e-5, id="cpu-f32"),
    pytest.param("cuda", F64, 1e-12, id="accel-f64", marks=on_accelerator),
    pytest.param("cuda", F32, 1e-5, id="accel-f32", marks=on_accelerator),
])
def test_collective_kernel(record_property, kernel, device, dtype, limit):
    ref = kernel("cpu", F64)
    worst = ((kernel(device, dtype) - ref).abs().amax(0) / ref.abs().amax(0)).max().item()
    record_property("relative_difference", worst)
    assert worst < limit
