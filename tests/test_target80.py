import sys,unittest
from unittest.mock import Mock,patch
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baseline'))
from target80_gate_search import candidates,evaluate,choose,codes_for
import target80_runtime as runtime

class Target80Tests(unittest.TestCase):
    def test_prefix_search_matches_direct_paths_including_ties(self):
        rng=np.random.default_rng(42)
        e=np.array([0.,.5,.5,1.,1.,2.]);n=rng.integers(0,2,6).astype(bool);w=rng.integers(0,2,(6,5)).astype(bool)
        for row in candidates(e,n,w,[0.,.5,1.,2.,np.inf]):
            correct,cost,_,_=evaluate(e,n,w,row)
            self.assertEqual((correct,cost),(row['correct'],row['cost']))

    def test_objectives_cannot_use_infeasible_candidate(self):
        rows=[{'correct':79,'cost':90},{'correct':80,'cost':105},{'correct':81,'cost':110},{'correct':82,'cost':200}]
        choice=choose(rows,100,100)
        self.assertEqual(choice['same_budget'],rows[0]);self.assertEqual(choice['target_80.0'],rows[1]);self.assertEqual(choice['target_80.2'],rows[2])

    def test_boundary_equality_uses_higher_compute_path(self):
        p={'kind':'monotone','gate':.5,'t1':1.,'t2':1.5}
        self.assertEqual(codes_for([.49,.5,.99,1.,1.5],p).tolist(),[-1,0,0,1,2])

    def test_numpy_and_runtime_boundaries_agree_without_rounding(self):
        e=torch.tensor([.5,1.,1.5],dtype=torch.float32)
        for p in [{'kind':'monotone','gate':.5+1e-9,'t1':1.+1e-9,'t2':'+inf'},
                  {'kind':'fixed','gate':1.,'ratio':.5}]:
            np.testing.assert_array_equal(runtime.torch_codes(e,p).numpy(),codes_for(e.numpy(),p))

    def test_actual_execution_visits_each_routed_image_once(self):
        model=Mock(training=False);model._stage_ratio.side_effect=lambda h:h/6
        model.entropy.return_value=torch.tensor([[1.5],[.7],[.9],[1.2]])
        images=torch.arange(4.)[:,None];features=images+100;logits=images.clone();att=torch.ones(4,197);calls=[]
        def wide(m,z,mask):calls.extend(z[:,0].tolist());return z+10+mask
        with patch.object(runtime,'forward_narrow_with_attention',return_value=(features,logits,att)), \
             patch.object(runtime,'wide_input',side_effect=lambda m,x,f:x), \
             patch.object(runtime,'make_update_mask',side_effect=lambda a,r,p:r), \
             patch.object(runtime,'wide_logits',side_effect=wide):
            out,codes,_=runtime.forward_target80(model,images,{'kind':'monotone','gate':.8,'t1':1.25,'t2':'+inf'})
        self.assertEqual(codes.tolist(),[1,-1,0,0]);self.assertEqual(sorted(calls),[0.,2.,3.])
        torch.testing.assert_close(out,torch.tensor([[10.5],[1.],[12.25],[13.25]]))

if __name__=='__main__':unittest.main()
