"""RTPLAN 내보내기 테스트.

핵심 검증: 쓴 파일을 **다시 읽어** dwell time이 살아남는가 (DS 문자열 왕복 포함),
TRAK가 정합적인가, 안전 태그가 강제되는가.
Geneva_Case가 없으면 전체 skip.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CASE = os.path.join(ROOT, "Geneva_Case")
HAS_CASE = os.path.isdir(CASE) and any(f.endswith(".dcm") for f in os.listdir(CASE))

if HAS_CASE:
    import pydicom

    from brachyplan.io import find_files, load_plan
    from brachyplan.io.rtplan_writer import (
        _ds, quantize_times, short_uid, uid_root_of, verify_rtplan, write_rtplan,
    )


class TestDSFormatting(unittest.TestCase):
    """DICOM DS VR은 16자 제한 — 넘으면 파일이 규격 위반이 된다."""

    @unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
    def test_within_16_chars(self):
        for v in [0.0, 1.0, 245.0, 0.0728325165716671, 1e-9, 1234567.891234567,
                  2736.23246450696, 1e12]:
            self.assertLessEqual(len(_ds(v)), 16, f"{v} -> {_ds(v)!r}")

    @unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
    def test_roundtrip_precision(self):
        for v in [0.1, 70.0, 39.3, 0.0728325165716671]:
            self.assertAlmostEqual(float(_ds(v)), v, places=9)

    @unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
    def test_zero(self):
        self.assertEqual(float(_ds(0.0)), 0.0)


class TestQuantize(unittest.TestCase):
    @unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
    def test_rounds_to_resolution(self):
        t = quantize_times([1.04, 1.06, 0.0, 9.99], 0.1)
        np.testing.assert_allclose(t, [1.0, 1.1, 0.0, 10.0], atol=1e-9)

    @unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
    def test_no_quantize_when_zero(self):
        t = quantize_times([1.234567], 0.0)
        self.assertAlmostEqual(t[0], 1.234567, places=9)

    @unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
    def test_never_negative(self):
        self.assertTrue(np.all(quantize_times([-5.0, 0.02], 0.1) >= 0))


@unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
class TestWriteRTPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = find_files(CASE)["RTPLAN"][0]
        cls.plan = load_plan(cls.src)
        cls.n = len(cls.plan.dwells)
        cls.t_orig = np.array([d.time_s for d in cls.plan.dwells])
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _out(self, name="out.dcm"):
        return os.path.join(self.tmp, name)

    # ------------------------------------------------------------------
    def test_roundtrip_identity(self):
        """원본 시간을 그대로 써서 다시 읽으면 원본과 같아야 한다."""
        out = self._out("identity.dcm")
        write_rtplan(self.src, out, self.t_orig, time_resolution_s=0.0)
        back = np.array([d.time_s for d in load_plan(out).dwells])
        np.testing.assert_allclose(back, self.t_orig, atol=1e-9)

    def test_roundtrip_new_times(self):
        rng = np.random.default_rng(0)
        t = np.round(rng.uniform(0, 40, self.n), 1)
        out = self._out("new.dcm")
        write_rtplan(self.src, out, t, time_resolution_s=0.1)
        ok, msgs = verify_rtplan(out, t)
        self.assertTrue(ok, "\n".join(msgs))

    def test_positions_unchanged(self):
        """dwell position은 절대 변하지 않아야 한다."""
        out = self._out("pos.dcm")
        write_rtplan(self.src, out, np.full(self.n, 3.0))
        p0 = np.array([d.position for d in self.plan.dwells])
        p1 = np.array([d.position for d in load_plan(out).dwells])
        np.testing.assert_allclose(p1, p0, atol=1e-9)

    def test_trak_recomputed(self):
        out = self._out("trak.dcm")
        t = np.full(self.n, 2.0)
        r = write_rtplan(self.src, out, t, time_resolution_s=0.0)
        sk = float(pydicom.dcmread(self.src).SourceSequence[0].ReferenceAirKermaRate)
        self.assertAlmostEqual(r.trak, sk * t.sum() / 3600.0, places=6)
        ds = pydicom.dcmread(out)
        self.assertAlmostEqual(
            float(ds.ApplicationSetupSequence[0].TotalReferenceAirKerma),
            r.trak, places=4)

    def test_channel_total_times(self):
        out = self._out("ct.dcm")
        t = np.full(self.n, 1.5)
        write_rtplan(self.src, out, t, time_resolution_s=0.0)
        ds = pydicom.dcmread(out)
        ch_of = np.array([d.channel for d in self.plan.dwells])
        for ch in ds.ApplicationSetupSequence[0].ChannelSequence:
            n_in_ch = int(np.sum(ch_of == int(ch.ChannelNumber)))
            self.assertAlmostEqual(float(ch.ChannelTotalTime), 1.5 * n_in_ch, places=6)

    # --- 안전 -----------------------------------------------------------
    def test_approval_forced_unapproved(self):
        """원본이 APPROVED여도 결과는 반드시 UNAPPROVED."""
        self.assertEqual(str(pydicom.dcmread(self.src).ApprovalStatus), "APPROVED")
        out = self._out("appr.dcm")
        write_rtplan(self.src, out, self.t_orig)
        self.assertEqual(str(pydicom.dcmread(out).ApprovalStatus), "UNAPPROVED")

    def test_review_info_removed(self):
        out = self._out("rev.dcm")
        write_rtplan(self.src, out, self.t_orig)
        ds = pydicom.dcmread(out)
        for tag in ("ReviewDate", "ReviewTime", "ReviewerName"):
            self.assertNotIn(tag, ds)

    def test_new_sop_and_series_uid(self):
        src_ds = pydicom.dcmread(self.src)
        out = self._out("uid.dcm")
        write_rtplan(self.src, out, self.t_orig)
        ds = pydicom.dcmread(out)
        self.assertNotEqual(ds.SOPInstanceUID, src_ds.SOPInstanceUID)
        self.assertNotEqual(ds.SeriesInstanceUID, src_ds.SeriesInstanceUID)
        self.assertEqual(ds.file_meta.MediaStorageSOPInstanceUID, ds.SOPInstanceUID)

    # --- 수신 TPS 호환 옵션 -------------------------------------------
    def test_keep_series_uid(self):
        src_ds = pydicom.dcmread(self.src)
        out = self._out("keepser.dcm")
        write_rtplan(self.src, out, self.t_orig, keep_series_uid=True)
        ds = pydicom.dcmread(out)
        self.assertEqual(ds.SeriesInstanceUID, src_ds.SeriesInstanceUID)
        # SOP UID는 그래도 새로 발급되어야 한다 (별개 인스턴스)
        self.assertNotEqual(ds.SOPInstanceUID, src_ds.SOPInstanceUID)

    def test_uid_style_source_keeps_root_and_length(self):
        """구형 수신 시스템 대비 — 원본 UID 루트와 길이대(<=64)를 따른다."""
        src_ds = pydicom.dcmread(self.src)
        out = self._out("uidsrc.dcm")
        write_rtplan(self.src, out, self.t_orig, uid_style="source")
        ds = pydicom.dcmread(out)
        root = uid_root_of(str(src_ds.SOPInstanceUID))
        self.assertTrue(str(ds.SOPInstanceUID).startswith(root + "."))
        self.assertLessEqual(len(str(ds.SOPInstanceUID)),
                             len(str(src_ds.SOPInstanceUID)))
        self.assertNotEqual(ds.SOPInstanceUID, src_ds.SOPInstanceUID)

    def test_uid_style_pydicom_is_default_64(self):
        out = self._out("uidpyd.dcm")
        write_rtplan(self.src, out, self.t_orig)
        ds = pydicom.dcmread(out)
        self.assertLessEqual(len(str(ds.SOPInstanceUID)), 64)

    def test_uid_style_invalid_raises(self):
        with self.assertRaises(ValueError):
            write_rtplan(self.src, self._out("bad.dcm"), self.t_orig,
                         uid_style="nonsense")

    def test_new_private_uid(self):
        tag = pydicom.tag.Tag(0x0021, 0x1000)
        src_ds = pydicom.dcmread(self.src)
        if tag not in src_ds:
            self.skipTest("이 케이스에는 私有 태그 (0021,1000)이 없습니다")
        out = self._out("privuid.dcm")
        write_rtplan(self.src, out, self.t_orig, new_private_uid=True)
        ds = pydicom.dcmread(out)
        self.assertNotEqual(str(ds[tag].value), str(src_ds[tag].value))
        # 기본값에서는 건드리지 않는다
        out2 = self._out("privuid2.dcm")
        write_rtplan(self.src, out2, self.t_orig)
        self.assertEqual(str(pydicom.dcmread(out2)[tag].value),
                         str(src_ds[tag].value))

    def test_compat_options_preserve_geometry(self):
        """어떤 호환 옵션을 켜도 dwell 위치는 바뀌지 않아야 한다."""
        base = np.array([d.position for d in self.plan.dwells])
        for kw in (dict(keep_series_uid=True),
                   dict(uid_style="source"),
                   dict(new_private_uid=True),
                   dict(uid_style="source", new_private_uid=True,
                        keep_series_uid=True)):
            out = self._out(f"geo_{abs(hash(str(kw)))}.dcm")
            write_rtplan(self.src, out, self.t_orig, **kw)
            got = np.array([d.position for d in load_plan(out).dwells])
            np.testing.assert_array_equal(got, base, err_msg=str(kw))

    def test_study_and_frame_of_reference_preserved(self):
        """같은 검사·좌표계여야 구조물/영상과 함께 쓸 수 있다."""
        src_ds = pydicom.dcmread(self.src)
        out = self._out("for.dcm")
        write_rtplan(self.src, out, self.t_orig)
        ds = pydicom.dcmread(out)
        self.assertEqual(ds.StudyInstanceUID, src_ds.StudyInstanceUID)
        self.assertEqual(ds.FrameOfReferenceUID, src_ds.FrameOfReferenceUID)
        self.assertEqual(ds.PatientID, src_ds.PatientID)

    def test_label_marked(self):
        out = self._out("label.dcm")
        write_rtplan(self.src, out, self.t_orig, label_suffix="_AI")
        ds = pydicom.dcmread(out)
        self.assertIn("_AI", str(ds.RTPlanLabel))
        self.assertLessEqual(len(str(ds.RTPlanLabel)), 16)      # SH VR
        self.assertIn("RESEARCH", str(ds.RTPlanDescription).upper())

    def test_source_sequence_untouched(self):
        src_ds = pydicom.dcmread(self.src)
        out = self._out("src.dcm")
        write_rtplan(self.src, out, self.t_orig)
        a = src_ds.SourceSequence[0]
        b = pydicom.dcmread(out).SourceSequence[0]
        self.assertEqual(float(a.ReferenceAirKermaRate), float(b.ReferenceAirKermaRate))
        self.assertEqual(a.SourceStrengthReferenceDate, b.SourceStrengthReferenceDate)
        self.assertEqual(a.SourceIsotopeName, b.SourceIsotopeName)

    # --- 경계 조건 -------------------------------------------------------
    def test_empty_channel_keeps_dwell_count(self):
        """어떤 채널이 전부 0이 되어도 재읽기 시 dwell 개수가 유지돼야 한다.

        (로더가 ChannelTotalTime==0 채널을 건너뛰면 개수가 어긋난다 — 실제 버그였음)
        """
        ch_of = np.array([d.channel for d in self.plan.dwells])
        t = self.t_orig.copy()
        t[ch_of == 2] = 0.0
        out = self._out("empty.dcm")
        r = write_rtplan(self.src, out, t, time_resolution_s=0.1)
        back = load_plan(out)
        self.assertEqual(len(back.dwells), self.n)
        self.assertIn(2, r.empty_channels)
        self.assertTrue(any("전부 0" in w for w in r.warnings))
        ok, msgs = verify_rtplan(out, quantize_times(t, 0.1))
        self.assertTrue(ok, "\n".join(msgs))

    def test_empty_channel_avoids_zero_final_weight(self):
        """0/0 을 피하려 FinalCumulativeTimeWeight를 1로 둔다."""
        ch_of = np.array([d.channel for d in self.plan.dwells])
        t = self.t_orig.copy()
        t[ch_of == 2] = 0.0
        out = self._out("empty2.dcm")
        write_rtplan(self.src, out, t)
        for ch in pydicom.dcmread(out).ApplicationSetupSequence[0].ChannelSequence:
            self.assertGreater(float(ch.FinalCumulativeTimeWeight), 0.0)

    def test_all_zero_times(self):
        out = self._out("allzero.dcm")
        r = write_rtplan(self.src, out, np.zeros(self.n))
        self.assertEqual(r.total_time_s, 0.0)
        self.assertEqual(r.trak, 0.0)
        self.assertEqual(len(load_plan(out).dwells), self.n)

    def test_wrong_dwell_count_raises(self):
        with self.assertRaises(ValueError) as cm:
            write_rtplan(self.src, self._out("bad.dcm"), np.ones(self.n + 1))
        self.assertIn("불일치", str(cm.exception))

    def test_non_brachy_plan_raises(self):
        """일반 EBRT RTPLAN을 주면 명확히 거부해야 한다."""
        ds = pydicom.dcmread(self.src)
        del ds.ApplicationSetupSequence
        fake = self._out("ebrt.dcm")
        ds.save_as(fake)
        with self.assertRaises(ValueError) as cm:
            write_rtplan(fake, self._out("x.dcm"), self.t_orig)
        self.assertIn("brachytherapy", str(cm.exception))

    # --- 검증 함수 자체 ---------------------------------------------------
    def test_verify_detects_tampered_approval(self):
        out = self._out("tamper.dcm")
        write_rtplan(self.src, out, self.t_orig)
        ds = pydicom.dcmread(out)
        ds.ApprovalStatus = "APPROVED"
        ds.save_as(out)
        ok, msgs = verify_rtplan(out, self.t_orig)
        self.assertFalse(ok)
        self.assertTrue(any("UNAPPROVED" in m for m in msgs))

    def test_verify_detects_wrong_trak(self):
        out = self._out("badtrak.dcm")
        write_rtplan(self.src, out, self.t_orig)
        ds = pydicom.dcmread(out)
        ds.ApplicationSetupSequence[0].TotalReferenceAirKerma = "1.0"
        ds.save_as(out)
        ok, msgs = verify_rtplan(out, self.t_orig)
        self.assertFalse(ok)
        self.assertTrue(any("TRAK" in m for m in msgs))

    def test_verify_detects_time_mismatch(self):
        out = self._out("mismatch.dcm")
        write_rtplan(self.src, out, self.t_orig)
        ok, _ = verify_rtplan(out, self.t_orig + 1.0)
        self.assertFalse(ok)


@unittest.skipUnless(HAS_CASE, "Geneva_Case 없음")
class TestDosePointRecalc(unittest.TestCase):
    def test_dose_points_updated_when_engine_given(self):
        from brachyplan.dose import SourceData, TG43Calculator
        src = find_files(CASE)["RTPLAN"][0]
        plan = load_plan(src)
        calc = TG43Calculator(SourceData.load("ir192_mHDR_v2_clrp"))
        tmp = tempfile.mkdtemp()
        try:
            out = os.path.join(tmp, "dp.dcm")
            t = np.array([d.time_s for d in plan.dwells])
            r = write_rtplan(src, out, t, time_resolution_s=0.0,
                             recompute_dose_points_with=(calc, plan.sk_ref))
            # RTPLAN의 DoseReferenceSequence에 좌표가 없으면 경고만 남는다
            if r.dose_points_updated:
                for name, gy in r.dose_points_updated.items():
                    self.assertGreater(gy, 0.0)
                    self.assertIn(name, r.dose_points_original)
            else:
                self.assertTrue(any("좌표" in w for w in r.warnings))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestUidHelpers(unittest.TestCase):
    """UID 보조 함수 — 케이스 데이터가 없어도 돌아간다."""

    def test_uid_root_strips_three_components(self):
        from brachyplan.io.rtplan_writer import uid_root_of
        self.assertEqual(uid_root_of("1.3.6.1.4.1.2452.6.111.222.333.444"),
                         "1.3.6.1.4.1.2452.6.111")

    def test_uid_root_of_short_uid_is_identity(self):
        from brachyplan.io.rtplan_writer import uid_root_of
        self.assertEqual(uid_root_of("1.2.3.4"), "1.2.3.4")

    def test_short_uid_respects_max_len(self):
        from brachyplan.io.rtplan_writer import short_uid
        root = "1.3.6.1.4.1.2452.6.111"
        for n in (32, 45, 60, 64):
            u = short_uid(root, max_len=n)
            self.assertLessEqual(len(u), n)
            self.assertTrue(u.startswith(root + "."))

    def test_short_uid_no_leading_zero_in_last_component(self):
        from brachyplan.io.rtplan_writer import short_uid
        for _ in range(200):
            last = short_uid("1.2.3.4", max_len=40).split(".")[-1]
            self.assertNotEqual(last[0], "0")

    def test_short_uid_unique(self):
        from brachyplan.io.rtplan_writer import short_uid
        u = {short_uid("1.2.3.4", 60) for _ in range(500)}
        self.assertEqual(len(u), 500)

    def test_short_uid_root_too_long_raises(self):
        from brachyplan.io.rtplan_writer import short_uid
        with self.assertRaises(ValueError):
            short_uid("1" * 60, max_len=62)


if __name__ == "__main__":
    unittest.main(verbosity=2)
