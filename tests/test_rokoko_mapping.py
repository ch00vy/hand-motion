import copy
import json
from pathlib import Path
import socket
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from app.realtime.rokoko_geometry import hand_basis
from app.realtime.rokoko_mapping import RokokoHandMapper, hand_points, QPOS_LIMITS, FLEXION_TARGETS, raw_flexion_qpos
from app.realtime.rokoko_wrist_mapping import RokokoWristMapper, WristMappingConfig
from app.gui.recorder_controller import UdpQposProvider


def pose():
    points = {"wrist": np.zeros(3), "wrist_rotation": np.array([0., 0., 0., 1.])}
    for name, x in (("index", .03), ("middle", .01), ("ring", -.01), ("little", -.03), ("thumb", .06)):
        for i, segment in enumerate(("proximal", "medial", "distal", "tip")):
            points[f"{name}_{segment}"] = np.array([x, .08 + .03 * i, 0.])
    return points


def rigid(points, rotation, translation=None):
    result = {k: rotation.apply(v) + (np.zeros(3) if translation is None else translation)
              for k, v in points.items() if k != "wrist_rotation"}
    result["wrist_rotation"] = (rotation * Rotation.from_quat(points["wrist_rotation"])).as_quat()
    return result


def packet(points):
    body = {"rightHand": {"position": dict(zip("xyz", points["wrist"].tolist())),
                           "rotation": dict(zip("xyzw", points["wrist_rotation"].tolist()))}}
    for name in ("thumb", "index", "middle", "ring", "little"):
        for seg in ("proximal", "medial", "distal", "tip"):
            body[f"right{name.title()}{seg.title()}"] = {"position": dict(zip("xyz", points[f"{name}_{seg}"].tolist()))}
    return {"scene": {"actors": [{"body": body}]}}


class MappingTests(unittest.TestCase):
    def test_neutral_return_and_recalibration(self):
        mapper = RokokoHandMapper()
        p = rigid(pose(), Rotation.from_euler("xyz", [.7, -.3, .2]), np.array([2., 3., 4.]))
        assert_allclose(mapper.map_points(p), 0., atol=1e-8)
        moved = rigid(p, Rotation.from_rotvec([.1, .2, .15]))
        self.assertGreater(np.linalg.norm(mapper.map_points(moved)[:3]), .05)
        assert_allclose(mapper.map_points(p), 0., atol=1e-8)
        assert_allclose(mapper.calibrate_open(moved), 0., atol=1e-8)

    def test_wrist_axes_both_directions_and_world_invariance(self):
        p = pose()
        basis = hand_basis(p)
        global_rotation = Rotation.from_euler("xyz", [1.1, -.7, .4])
        for i, axis in enumerate(WristMappingConfig().axes):
            for angle in (-.1, .1):
                moved = rigid(p, Rotation.from_rotvec(basis @ (np.array(axis) * angle)))
                expected = np.zeros(3)
                expected[i] = -angle if i in (0, 2) else angle
                for source, target in ((p, moved), (rigid(p, global_rotation), rigid(moved, global_rotation))):
                    mapper = RokokoHandMapper()
                    mapper.calibrate_open(source)
                    assert_allclose(mapper.map_points(target)[:3], expected, atol=2e-5)

    def test_palm_direction_correction_preserves_other_joints_and_neutral(self):
        p = pose()
        corrected = RokokoHandMapper()
        canonical = RokokoHandMapper()
        canonical.wrist = RokokoWristMapper(WristMappingConfig(pronation_sign=1.0))
        for mapper in (corrected, canonical):
            mapper.calibrate_open(p)
        for angle in (-.7, .7):
            moved = rigid(p, Rotation.from_rotvec([.15, angle, -.08]))
            before = canonical.map_points(moved)
            after = corrected.map_points(moved)
            self.assertGreater(abs(before[0]), .4)
            assert_allclose(after[0], -before[0], atol=1e-8)
            assert_allclose(after[1:], before[1:], atol=1e-8)
        assert_allclose(corrected.map_points(p), 0., atol=1e-8)
        assert_allclose(corrected.calibrate_open(moved), 0., atol=1e-8)

    def test_deviation_direction_preserves_other_joints_and_full_range(self):
        p = pose()
        corrected = RokokoHandMapper()
        corrected.wrist = RokokoWristMapper(WristMappingConfig(deviation_sign=-1.0))
        previous = RokokoHandMapper()
        previous.wrist = RokokoWristMapper(WristMappingConfig(deviation_sign=1.0))
        for mapper in (corrected, previous):
            mapper.calibrate_open(p)
        for angle in (-.1, .1):
            moved = rigid(p, Rotation.from_rotvec([.08, .12, angle]))
            before = previous.map_points(moved)
            after = corrected.map_points(moved)
            assert_allclose(after[1], -before[1], atol=1e-6)
            assert_allclose(after[[0, 2]], before[[0, 2]], atol=1e-6)
            assert_allclose(after[3:], before[3:], atol=1e-8)
        # +0.35 output is allowed even though the opposite model limit is -0.1745.
        axis = np.asarray(WristMappingConfig().axes[1])
        moved = rigid(p, Rotation.from_rotvec(hand_basis(p) @ (axis * -.35)))
        assert_allclose(corrected.map_points(moved)[1], .35, atol=2e-5)
        assert_allclose(corrected.map_points(p), 0., atol=1e-8)

    def test_palm_bending_correction_keeps_sideways_deviation_and_twist(self):
        p = pose()
        corrected = RokokoHandMapper()
        previous = RokokoHandMapper()
        previous.wrist = RokokoWristMapper(WristMappingConfig(flexion_sign=1.0))
        for mapper in (corrected, previous):
            mapper.calibrate_open(p)
        basis = hand_basis(p)
        axes = np.asarray(WristMappingConfig().axes)
        for bend in (-.4, .4):
            # Include twist and sideways motion to detect accidental flips of
            # either of the other wrist coordinates during palm bending.
            rotation = Rotation.identity()
            for axis, angle in zip(axes, (.15, .1, bend)):
                rotation = rotation * Rotation.from_rotvec(basis @ (axis * angle))
            moved = rigid(p, rotation)
            before = previous.map_points(moved)
            after = corrected.map_points(moved)
            assert_allclose(after[2], -bend, atol=2e-5)
            assert_allclose(after[[0, 1]], [-.15, .1], atol=2e-5)
            assert_allclose(after[[0, 1]], before[[0, 1]], atol=1e-6)
            assert_allclose(after[3:], before[3:], atol=1e-8)
        assert_allclose(corrected.map_points(p), 0., atol=1e-8)
        assert_allclose(corrected.calibrate_open(moved), 0., atol=1e-8)

    def test_spread_is_not_flexion(self):
        p = pose()
        mapper = RokokoHandMapper()
        mapper.calibrate_open(p)
        spread = copy.deepcopy(p)
        for seg in ("medial", "distal", "tip"):
            vector = p[f"index_{seg}"] - p["index_proximal"]
            spread[f"index_{seg}"] = p["index_proximal"] + Rotation.from_euler("z", -.2).apply(vector)
        q = mapper.map_points(spread)
        assert_allclose(q[8], .2, atol=1e-8)
        assert_allclose(q[[7, 9, 10]], 0., atol=1e-8)

    def test_flexion_is_not_spread_and_fist_is_optional(self):
        p = pose()
        mapper = RokokoHandMapper()
        mapper.calibrate_open(p)
        bent = copy.deepcopy(p)
        for name in ("index", "thumb"):
            for seg in ("medial", "distal", "tip"):
                bent[f"{name}_{seg}"] = p[f"{name}_proximal"] + Rotation.from_euler("x", .3).apply(p[f"{name}_{seg}"] - p[f"{name}_proximal"])
        direct = mapper.map_points(bent)
        assert_allclose(direct[7], .3, atol=1e-8)
        assert_allclose(direct[8], 0., atol=1e-8)
        self.assertTrue(mapper.calibrate_closed()[0])
        scaled = mapper.map_points(bent)
        assert_allclose(scaled[7], QPOS_LIMITS[7, 1], atol=1e-8)
        # Uncalibrated joints retain their direct mapping.
        assert_allclose(scaled[5:7], direct[5:7])
        assert_allclose(mapper.map_points(p), 0., atol=1e-8)
        mapper.calibrate_open(p)
        self.assertIsNone(mapper.closed_flexion)

    def test_calibrated_fist_reaches_all_finger_limits_and_scales_smoothly(self):
        neutral = pose()

        def bend(amount):
            result = copy.deepcopy(neutral)
            for name in ("index", "middle", "ring", "little"):
                start = neutral[f"{name}_proximal"]
                angle = 0.
                for segment, delta in zip(("medial", "distal", "tip"), (.3, .4, .25)):
                    angle += delta * amount
                    start = start + .03 * np.array([0., np.cos(angle), np.sin(angle)])
                    result[f"{name}_{segment}"] = start
            return result

        mapper = RokokoHandMapper()
        mapper.calibrate_open(neutral)
        closed = bend(1.)
        direct = mapper.map_points(closed)
        success, message = mapper.calibrate_closed()
        self.assertTrue(success)
        self.assertIn("12/15", message)
        indices = [base + offset for base in (7, 11, 15, 19) for offset in (0, 2, 3)]
        maximum = mapper.map_points(closed)
        assert_allclose(maximum[indices], QPOS_LIMITS[indices, 1], atol=1e-8)
        self.assertTrue(np.all(maximum[indices] > direct[indices]))
        halfway = mapper.map_points(bend(.5))
        assert_allclose(halfway[indices], .5 * QPOS_LIMITS[indices, 1], atol=1e-8)
        assert_allclose(mapper.map_points(bend(1.2))[indices], maximum[indices], atol=1e-8)
        assert_allclose(mapper.map_points(neutral), 0., atol=1e-8)
        assert_allclose(maximum[:3], direct[:3], atol=1e-8)
        self.assertEqual(FLEXION_TARGETS[4], QPOS_LIMITS[4, 1])
        self.assertEqual(FLEXION_TARGETS[5], QPOS_LIMITS[5, 0])
        self.assertEqual(FLEXION_TARGETS[6], QPOS_LIMITS[6, 0])

    def test_negative_mcp_motion_is_calibrated_instead_of_clipped_away(self):
        neutral = pose()

        def curled(amount):
            result = copy.deepcopy(neutral)
            for name in ("index", "middle", "ring", "little"):
                start = neutral[f"{name}_proximal"]
                angle = 0.
                for segment, delta in zip(("medial", "distal", "tip"), (-1.4, -.4, -.25)):
                    angle += delta * amount
                    start = start + .03 * np.array([0., np.cos(angle), np.sin(angle)])
                    result[f"{name}_{segment}"] = start
            # Also move all three thumb flexion coordinates so this reproduces
            # the reported 11/15 calibration with four missing MCP joints.
            start = neutral["thumb_proximal"]
            for segment, angle in zip(("medial", "distal", "tip"), (.5, .9, 1.2)):
                start = start + .03 * np.array([0., np.cos(angle * amount), np.sin(angle * amount)])
                result[f"thumb_{segment}"] = start
            return result

        mapper = RokokoHandMapper()
        mapper.calibrate_open(neutral)
        closed = curled(1.)
        raw = raw_flexion_qpos(closed, neutral)
        mcp_indices = [7, 11, 15, 19]
        assert_allclose(raw[mcp_indices], -1.4, atol=1e-8)
        # Previously clipping these four values yielded 11/15 joints.
        old_count = sum(np.clip(raw, QPOS_LIMITS[:, 0], QPOS_LIMITS[:, 1])[i] * np.sign(t) >= .05
                        for i, t in FLEXION_TARGETS.items())
        self.assertEqual(old_count, 11)
        mapper.map_points(closed)
        success, message = mapper.calibrate_closed()
        self.assertTrue(success)
        self.assertIn("15/15", message)
        result = mapper.map_points(closed)
        for index, target in FLEXION_TARGETS.items():
            assert_allclose(result[index], target, atol=1e-8)
        assert_allclose(mapper.map_points(curled(.5))[mcp_indices], .5 * QPOS_LIMITS[mcp_indices, 1], atol=1e-8)
        # More bending past 90 degrees must not appear straighter again.
        assert_allclose(mapper.map_points(curled(1.2))[mcp_indices], QPOS_LIMITS[mcp_indices, 1], atol=1e-8)
        # Motion opposite the calibrated direction must remain open.
        assert_allclose(mapper.map_points(curled(-.4))[mcp_indices], 0., atol=1e-8)
        assert_allclose(mapper.map_points(neutral), 0., atol=1e-8)
        mapper.calibrate_open(neutral)
        self.assertIsNone(mapper.closed_flexion)

    def test_thumb_flexion_sign_before_and_after_scaling(self):
        p = pose()
        bent = copy.deepcopy(p)
        bent["thumb_distal"] = bent["thumb_medial"] + np.array([0., .03 * np.cos(.3), .03 * np.sin(.3)])
        bent["thumb_tip"] = bent["thumb_distal"] + np.array([0., .03 * np.cos(.6), .03 * np.sin(.6)])
        mapper = RokokoHandMapper()
        mapper.calibrate_open(p)
        q = mapper.map_points(bent)
        assert_allclose(q[5:7], [-.3, -.3], atol=1e-8)
        self.assertTrue(mapper.calibrate_closed()[0])
        self.assertTrue(np.all(mapper.map_points(bent)[5:7] < 0.))

    def test_packet_validation(self):
        p = packet(pose())
        hand_points(p)
        p["scene"]["actors"][0]["body"]["rightHand"]["rotation"] = dict.fromkeys("xyzw", 0.)
        with self.assertRaises(ValueError):
            hand_points(p)

    def test_udp_direct_and_published_qpos_and_stationary_frames(self):
        provider = UdpQposProvider(port=0)
        self.assertTrue(provider.start())
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            destination = provider.socket.getsockname()
            p = pose()
            sender.sendto(json.dumps(packet(p)).encode(), destination)
            provider.latest_qpos()
            self.assertTrue(provider.calibrate_rokoko_open()[0])
            moved = rigid(p, Rotation.from_rotvec([.12, .04, -.06]))
            sender.sendto(json.dumps(packet(moved)).encode(), destination)
            direct = provider.latest_qpos()
            publisher_mapper = RokokoHandMapper()
            publisher_mapper.map_points(hand_points(packet(p)))
            published = publisher_mapper.map_points(hand_points(packet(moved)))
            assert_allclose(direct, published)
            provider._last_qpos_change_time -= 5.
            self.assertFalse(provider.status().stuck)
            sender.sendto(json.dumps({"type": "hand_qpos", "qpos": published.tolist()}).encode(), destination)
            assert_allclose(provider.latest_qpos(), direct)
            self.assertFalse(provider.calibrate_rokoko_open()[0])
            provider._last_qpos_sample_time -= 5.
            self.assertTrue(provider.status().stuck)
        finally:
            sender.close()
            provider.close()

    def test_limits_match_installed_mujoco_model(self):
        import mujoco
        path = Path("venv/Lib/site-packages/myosuite/simhive/myo_sim/hand/myohand.xml")
        if not path.exists():
            self.skipTest("Local MyoHand model not installed")
        model = mujoco.MjModel.from_xml_path(str(path))
        assert_allclose(QPOS_LIMITS, model.jnt_range)
        data = mujoco.MjData(model)

        def model_points():
            mujoco.mj_forward(model, data)
            result = {"wrist": data.xpos[model.body("lunate").id].copy(),
                      "wrist_rotation": Rotation.from_matrix(data.xmat[model.body("lunate").id].reshape(3, 3)).as_quat()}
            for name, number in (("index", 2), ("middle", 3), ("ring", 4), ("little", 5), ("thumb", 1)):
                joints = (["cmc_flexion", "mp_flexion", "ip_flexion"] if number == 1 else
                          [f"mcp{number}_flexion", f"pm{number}_flexion", f"md{number}_flexion"])
                for segment, joint in zip(("proximal", "medial", "distal"), joints):
                    result[f"{name}_{segment}"] = data.xanchor[model.joint(joint).id].copy()
                result[f"{name}_tip"] = result[f"{name}_distal"] + (result[f"{name}_distal"] - result[f"{name}_medial"]) / 2
            return result

        neutral = model_points()
        mapper = RokokoHandMapper()
        # Validate the geometric fit in the model's own rotation convention.
        # Live Rokoko direction corrections are tested separately above.
        mapper.wrist = RokokoWristMapper(WristMappingConfig(pronation_sign=1.0, deviation_sign=1.0, flexion_sign=1.0))
        mapper.calibrate_open(neutral)
        # Obtain the target rotation from actual MuJoCo forward kinematics,
        # independently of the mapper's stored hinge axes.
        for angles in ([.4, .2, -.3], [-.5, -.1, .4], [1., .3, .5]):
            data.qpos[:3] = angles
            result = mapper.map_points(model_points())
            assert_allclose(result[:3], angles, atol=2e-5)
        data.qpos[:] = 0.
        data.qpos[7] = .3
        result = mapper.map_points(model_points())
        self.assertGreater(result[7], .25)
        data.qpos[:] = 0.
        data.qpos[8] = .2
        result = mapper.map_points(model_points())
        assert_allclose(result[8], .2, atol=.015)


if __name__ == "__main__":
    unittest.main()
