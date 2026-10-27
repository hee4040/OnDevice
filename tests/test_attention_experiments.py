import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baseline'))
from analyze_attention import patch_metrics
from occlude_attention import masked_batch, effect_label


class AttentionExperimentTests(unittest.TestCase):
    def test_cls_excluded_and_patch_distribution_renormalized(self):
        a=np.ones((2,197));a[0,0]=100;a[1,0]=.001
        p,h,t=patch_metrics(a)
        np.testing.assert_allclose(p.sum(1),1)
        np.testing.assert_allclose(h,np.log(196))
        np.testing.assert_allclose(t,10/196)

    def test_one_patch_and_zero_patch_mass(self):
        a=np.zeros((1,197));a[0,0]=.5;a[0,196]=.5
        p,h,t=patch_metrics(a)
        np.testing.assert_allclose(h,0);np.testing.assert_allclose(t,1)
        a[0,196]=0
        with self.assertRaises(ValueError):patch_metrics(a)

    def test_masks_are_independent_and_spatially_aligned(self):
        x=torch.ones(3,224,224)
        b=masked_batch(x,[0,13,14,195])
        torch.testing.assert_close(b[0],x);torch.testing.assert_close(b[1],x)
        self.assertEqual(b.shape,(6,3,224,224))
        for i,(y,z) in enumerate([(0,0),(0,13),(1,0),(13,13)],2):
            self.assertEqual((b[i]==0).sum().item(),3*16*16)
            self.assertTrue((b[i,:,y*16:(y+1)*16,z*16:(z+1)*16]==0).all())
        self.assertTrue((x==1).all())

    def test_signed_margin_effect_and_numerical_tolerance(self):
        self.assertEqual(effect_label(-.2),'supports_top1_margin')
        self.assertEqual(effect_label(.2),'opposes_top1_margin')
        self.assertEqual(effect_label(1e-6),'near_zero')
        self.assertEqual(effect_label(0),'near_zero')


if __name__=='__main__':unittest.main()
