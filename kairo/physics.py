"""Physical simulation models for speedcubing dynamics and robot secondary motion.

Includes:
  1. Magnetic speedcube layer flicks: jerk-limited finger acceleration -> free coast
     against Coulomb + viscous friction -> nonlinear magnetic detent capture with
     authentic ~2.5-3.5 deg overshoot and damped settle.
  2. Double-flicks (180 deg turns) and minimum-jerk wrist turns with detent snap.
  3. 3-axis compliant grip recoil (reaction torque transmitted to whole cube & hands).
  4. Critically damped 2nd-order follower for head tracking, saccades, and torso lean.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from typing import Callable, List, Sequence, Tuple

QUARTER = 0.5 * math.pi


def min_jerk(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def min_jerk_d(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return 30.0 * s * s * (1.0 - s) ** 2


def min_jerk_dd(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return 60.0 * s * (1.0 - s) * (1.0 - 2.0 * s)


class Profile:
    """Sampled scalar trajectory y(t) with fast interpolation, velocity & acceleration."""

    def __init__(self, ts: List[float], ys: List[float], capture_time: float = 0.0) -> None:
        self.ts = ts
        self.ys = ys
        self.duration = ts[-1]
        self.capture_time = capture_time  # exact instant magnetic detent clicks home

    def __call__(self, t: float) -> float:
        if t <= 0.0:
            return self.ys[0]
        if t >= self.duration:
            return self.ys[-1]
        i = bisect_left(self.ts, t)
        t0, t1 = self.ts[i - 1], self.ts[i]
        a = (t - t0) / (t1 - t0)
        return self.ys[i - 1] * (1.0 - a) + self.ys[i] * a

    def vel(self, t: float, h: float = 5e-4) -> float:
        return (self(t + h) - self(t - h)) / (2.0 * h)

    def accel(self, t: float, h: float = 1e-3) -> float:
        return (self(t + h) - 2.0 * self(t) + self(t - h)) / (h * h)


# ------------------------------------------------------------------ speedcube layer physics
# Tuned to a modern magnetic 57mm speedcube (GAN 12 / MoYu WRM feel)
DETENT_K = 46000.0            # rad/s^2 per rad peak magnet stiffness
DETENT_W = math.radians(7.8)  # angular capture width of neodymium edge/corner magnets
COULOMB = 560.0               # rad/s^2 dry friction
VISCOUS = 13.5                # 1/s silicone lube damping
DETENT_EXTRA_DAMP = 58.0      # extra eddy/bump damping inside detent well
DT = 1e-4                     # 0.1 ms RK2 integration step


def _detent_acc(theta: float) -> float:
    d = theta - QUARTER * round(theta / QUARTER)
    return -DETENT_K * d * math.exp(-((d / DETENT_W) ** 2))


def _simulate_flick(theta_rel: float, w_rel: float, t_max: float = 0.24):
    """Smooth sinusoidal-pulse finger acceleration to (theta_rel, w_rel), then free coast + detent."""
    # For a half-sine acceleration pulse a(t) = A*sin(pi*t/Td):
    #   w(Td) = 2*A*Td/pi = w_rel  -> A = pi*w_rel/(2*Td)
    #   theta(Td) = w_rel * Td / 2 = theta_rel -> Td = 2*theta_rel / w_rel
    T_d = 2.0 * theta_rel / w_rel
    A = 0.5 * math.pi * w_rel / T_d
    ts: List[float] = []
    th: List[float] = []
    t = 0.0
    while t < T_d:
        s = t / T_d
        # exact integral of A*sin(pi*s):
        #   w(t) = (A*Td/pi) * (1 - cos(pi*s))
        #   theta(t) = (A*Td/pi) * (t - (Td/pi)*sin(pi*s))
        theta = (w_rel * 0.5) * (t - (T_d / math.pi) * math.sin(math.pi * s))
        ts.append(t)
        th.append(theta)
        t += DT

    theta, w = theta_rel, w_rel
    peak = theta
    capture_t = None
    settled_for = 0.0
    while t < T_d + t_max:
        def rhs(th_val: float, w_val: float) -> float:
            dd = th_val - QUARTER * round(th_val / QUARTER)
            visc = VISCOUS + (DETENT_EXTRA_DAMP if abs(dd) < DETENT_W else 0.0)
            det = _detent_acc(th_val)
            if abs(w_val) < 1e-3 and abs(det) < COULOMB:
                return 0.0
            fric = COULOMB * math.copysign(1.0, w_val) if abs(w_val) > 1e-3 else 0.0
            return det - visc * w_val - fric

        # Midpoint (RK2) integration for accuracy
        a1 = rhs(theta, w)
        if a1 == 0.0 and abs(w) < 1e-3:
            w = 0.0
        else:
            th_mid = theta + 0.5 * w * DT
            w_mid = w + 0.5 * a1 * DT
            a2 = rhs(th_mid, w_mid)
            w += a2 * DT
            theta += w_mid * DT

        if capture_t is None and theta >= QUARTER:
            capture_t = t
        peak = max(peak, theta)
        ts.append(t)
        th.append(theta)

        if w == 0.0 or (abs(theta - QUARTER) < math.radians(0.25) and abs(w) < 0.25):
            settled_for += DT
            if settled_for > 0.004:
                break
        else:
            settled_for = 0.0
        t += DT

    if capture_t is None:
        capture_t = ts[-1]
    return ts, th, peak, T_d, capture_t


def flick_profile(theta_rel_deg: float = 42.0, margin: float = 1.07) -> Tuple[Profile, float]:
    """Generate a physically integrated quarter-turn flick profile.

    Returns (Profile, release_time_T_d).
    Peak overshoot is naturally ~2.5-3.5 deg before magnetic settle to exact 90.0 deg.
    """
    theta_rel = math.radians(theta_rel_deg)
    lo, hi = 6.0, 120.0
    for _ in range(38):
        mid = 0.5 * (lo + hi)
        _, th, _, _, _ = _simulate_flick(theta_rel, mid)
        if th[-1] < QUARTER - math.radians(5.0):
            lo = mid
        else:
            hi = mid

    ts, th, peak, T_d, cap_t = _simulate_flick(theta_rel, hi * margin)
    # Smooth final micro-settle to exact pi/2 over 10 ms
    err = th[-1] - QUARTER
    t_end = ts[-1]
    for k in range(1, 101):
        ts.append(t_end + k * DT)
        th.append(QUARTER + err * (1.0 - min_jerk(k / 100.0)))

    # Decimate cleanly to 0.5 ms while preserving exact endpoints
    ts2, th2 = ts[::5], th[::5]
    if ts2[-1] != ts[-1]:
        ts2.append(ts[-1])
        th2.append(QUARTER)
    return Profile(ts2, th2, capture_time=cap_t), T_d


def wrist_profile(duration: float = 0.135, dt: float = 5e-4, overshoot_deg: float = 1.4) -> Profile:
    """Minimum-jerk wrist-driven 90 deg turn with subtle magnetic detent micro-overshoot."""
    n = int(math.ceil(duration / dt))
    ts = [i * duration / n for i in range(n + 1)]
    th: List[float] = []
    ov = math.radians(overshoot_deg)
    # Main min-jerk reaches 90 deg at 0.86*T, slight detent overshoot & settle in final 14%
    t_cap = 0.86 * duration
    for t in ts:
        if t <= t_cap:
            s = t / t_cap
            th.append(QUARTER * min_jerk(s))
        else:
            u = (t - t_cap) / (duration - t_cap)
            # Damped half-sine micro-overshoot that smoothly settles to exact QUARTER
            bump = ov * math.sin(math.pi * u) * math.exp(-3.2 * u) * (1.0 - u)
            th.append(QUARTER + bump)
    th[-1] = QUARTER
    return Profile(ts, th, capture_time=t_cap)


# ------------------------------------------------------------------ compliant grip recoil
def grip_recoil_3d(
    t_start: float,
    t_end: float,
    torque_vec_fn: Callable[[float], Tuple[float, float, float]],
    omega: float = 72.0,
    zeta: float = 0.52,
    gain: float = 0.026,
    dt: float = 5e-4,
) -> Tuple[Profile, Profile, Profile]:
    """3-axis spring-damper angular recoil of the held cube from equal-and-opposite layer torques."""
    n = int((t_end - t_start) / dt) + 2
    ts: List[float] = []
    px: List[float] = []
    py: List[float] = []
    pz: List[float] = []
    phi = [0.0, 0.0, 0.0]
    vel = [0.0, 0.0, 0.0]
    t = t_start
    for _ in range(n):
        ts.append(t - t_start)
        px.append(phi[0])
        py.append(phi[1])
        pz.append(phi[2])
        tau = torque_vec_fn(t)
        for k in range(3):
            acc = -gain * tau[k] - 2.0 * zeta * omega * vel[k] - (omega * omega) * phi[k]
            vel[k] += acc * dt
            phi[k] += vel[k] * dt
        t += dt
    return Profile(ts, px), Profile(ts, py), Profile(ts, pz)


# ------------------------------------------------------------------ critically damped follower
class CriticallyDampedSmoother:
    """2nd-order critically damped filter for organic head tracking, saccades, and torso lean."""

    def __init__(self, omega: float) -> None:
        self.omega = omega

    def filter_series(self, ts: Sequence[float], xs: Sequence):
        out = []
        is_vec = isinstance(xs[0], (list, tuple))
        x = list(xs[0]) if is_vec else float(xs[0])
        v = [0.0] * len(x) if is_vec else 0.0
        prev_t = ts[0]
        w = self.omega
        for t, target in zip(ts, xs):
            dt = max(0.0, t - prev_t)
            prev_t = t
            steps = max(1, int(dt * w / 0.18))
            h = dt / steps if dt > 0 else 0.0
            if is_vec:
                for k in range(len(x)):
                    tk = target[k]
                    for _ in range(steps):
                        a = w * w * (tk - x[k]) - 2.0 * w * v[k]
                        v[k] += a * h
                        x[k] += v[k] * h
                out.append(tuple(x))
            else:
                tk = float(target)
                for _ in range(steps):
                    a = w * w * (tk - x) - 2.0 * w * v
                    v += a * h
                    x += v * h
                out.append(x)
        return out


if __name__ == "__main__":
    pf, td = flick_profile()
    peak_deg = math.degrees(max(pf.ys))
    print(f"Flick profile: T_drive={td*1000:.1f} ms, capture={pf.capture_time*1000:.1f} ms, "
          f"total={pf.duration*1000:.1f} ms, overshoot={peak_deg - 90.0:.2f} deg")
    wp = wrist_profile()
    print(f"Wrist profile: capture={wp.capture_time*1000:.1f} ms, total={wp.duration*1000:.1f} ms, "
          f"overshoot={math.degrees(max(wp.ys)) - 90.0:.2f} deg")
