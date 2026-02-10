"""Tests that smb_utils works without imaging deps (monai, nibabel, pydicom).

These tests run in an environment where monai is NOT installed.
They verify that:
1. `import smb_utils` succeeds and core EHR functionality is available
2. Imaging modules load without crashing (monai deferred to call time)
3. Calling monai-dependent imaging functions raises a clear ImportError
4. Non-monai imaging utilities (extract_imaging_info, etc.) still work
"""

import inspect
import sys

import pytest


@pytest.fixture(autouse=True)
def _skip_if_monai_installed():
    """Skip all tests in this module if monai is installed."""
    try:
        import monai  # noqa: F401
        pytest.skip("monai is installed — these tests require monai to be absent")
    except ImportError:
        pass


@pytest.fixture()
def fresh_smb_utils():
    """Import smb_utils with a clean module cache."""
    # Clear any cached smb_utils modules so each test starts fresh.
    to_remove = [k for k in sys.modules if k.startswith("smb_utils")]
    for k in to_remove:
        del sys.modules[k]
    import smb_utils
    return smb_utils


class TestCoreImport:
    """Core package must be importable without monai/nibabel/pydicom."""

    def test_import_succeeds(self, fresh_smb_utils):
        """import smb_utils should not raise when imaging deps are missing."""
        assert fresh_smb_utils is not None

    def test_process_ehr_info_is_callable(self, fresh_smb_utils):
        """EHR functionality must always be available and callable."""
        assert hasattr(fresh_smb_utils, "process_ehr_info")
        assert callable(fresh_smb_utils.process_ehr_info)
        sig = inspect.signature(fresh_smb_utils.process_ehr_info)
        assert "df" in sig.parameters

    def test_all_exports_defined(self, fresh_smb_utils):
        """__all__ should list all public API names."""
        assert hasattr(fresh_smb_utils, "__all__")
        assert "process_ehr_info" in fresh_smb_utils.__all__
        assert "fetch_medical_volume" in fresh_smb_utils.__all__


class TestImagingModulesLoadable:
    """Imaging submodules should import without crashing even without monai."""

    def test_imaging_process_importable(self, fresh_smb_utils):
        from smb_utils import imaging_process
        # Module loaded — key functions should be defined.
        assert hasattr(imaging_process, "fetch_medical_volume")
        assert hasattr(imaging_process, "preprocess_image")
        assert hasattr(imaging_process, "download_medical_image")
        assert callable(imaging_process.fetch_medical_volume)

    def test_ct_slice_process_importable(self, fresh_smb_utils):
        from smb_utils import ct_slice_process
        assert hasattr(ct_slice_process, "process_ct_to_slices")
        assert callable(ct_slice_process.process_ct_to_slices)

    def test_imaging_text_pairs_importable(self, fresh_smb_utils):
        from smb_utils import imaging_text_pairs
        assert hasattr(imaging_text_pairs, "create_imaging_text_pairs")
        assert callable(imaging_text_pairs.create_imaging_text_pairs)


class TestMonaiDeferredToCallTime:
    """Functions that need monai should raise ImportError when called, not at import."""

    def test_get_ct_transforms_raises(self, fresh_smb_utils):
        from smb_utils.imaging_process import get_ct_transforms
        with pytest.raises(ImportError, match="MONAI"):
            get_ct_transforms()

    def test_get_mri_transforms_raises(self, fresh_smb_utils):
        from smb_utils.imaging_process import get_mri_transforms
        with pytest.raises(ImportError, match="MONAI"):
            get_mri_transforms()

    def test_get_pet_transforms_raises(self, fresh_smb_utils):
        from smb_utils.imaging_process import get_pet_transforms
        with pytest.raises(ImportError, match="MONAI"):
            get_pet_transforms()

    def test_get_xray_transforms_raises(self, fresh_smb_utils):
        from smb_utils.imaging_process import get_xray_transforms
        with pytest.raises(ImportError, match="MONAI"):
            get_xray_transforms()

    def test_preprocess_image_raises(self, fresh_smb_utils):
        from smb_utils.imaging_process import preprocess_image
        with pytest.raises((ImportError, ValueError)):
            # ValueError for bad modality, ImportError for missing monai
            preprocess_image("/fake/path.nii.gz", modality="CT")


class TestNonMonaiUtilitiesWork:
    """Imaging utilities that don't require monai should work fine."""

    def test_extract_imaging_info_empty_input(self, fresh_smb_utils):
        from smb_utils.imaging_process import extract_imaging_info
        result = extract_imaging_info([])
        assert result == []

    def test_extract_imaging_info_no_images(self, fresh_smb_utils):
        from smb_utils.imaging_process import extract_imaging_info
        conversations = [{"role": "user", "content": "hello"}]
        result = extract_imaging_info(conversations)
        assert result == []

    def test_extract_imaging_info_finds_image(self, fresh_smb_utils):
        from smb_utils.imaging_process import extract_imaging_info
        conversations = [[{
            "role": "user",
            "content": [{"image": "s3://bucket/scan.nii.gz", "modality": "CT"}],
        }]]
        result = extract_imaging_info(conversations)
        assert len(result) == 1
        assert result[0]["image"] == "s3://bucket/scan.nii.gz"
