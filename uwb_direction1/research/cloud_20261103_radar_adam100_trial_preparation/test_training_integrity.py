"""Synthetic/control-only fixtures; full100 loop, no real UWB, no native fitting."""
import unittest
from unittest.mock import patch
import native_adam100 as native
import adam_engine as engine

class TrainingControls(unittest.TestCase):
    def test_all100epochs23500callbacks_fixed_endpoint_no_best_selection(self):
        np=native.np;torch=native.torch
        class Model:
            def __init__(self):self.updates=0;self.training_calls=0
            def train(self):self.training_calls+=1
            def state_dict(self):return {'counter':torch.tensor(self.updates)}
        class Optimizer:
            def state_dict(self):return {'state':{},'param_groups':[]}
        model=Model();all_steps=[];epochs=[];positions=[]
        labels=np.arange(4700)%12;validation=np.arange(1000)%10
        def step(m,o,batch,y):
            self.assertEqual(batch.shape,(20,))
            np.testing.assert_array_equal(y,labels[batch])
            positions.extend(batch.tolist());m.updates+=1;return .25
        def evaluate(m,batch,y):
            np.testing.assert_array_equal(y,validation)
            # Deliberately make an early monitored score greater than final100.
            return np.zeros((1000,12),np.float32),1. if m.updates==470 else 0.
        def epoch(record,scores,state):
            n=record['epoch'];self.assertEqual(state['updates'],235*n)
            self.assertEqual(record['selection'],'fixed_final_epoch100_not_best_validation')
            self.assertFalse(record['diagnostic_labels_read'])
            self.assertFalse(record['users7_8_labels_used'])
            self.assertEqual(sorted(positions),(list(range(4700))));positions.clear()
            epochs.append(record.copy())
        with patch.object(native,'make',return_value=model),\
             patch.object(native,'optimizer',return_value=Optimizer()),\
             patch.object(native,'step',side_effect=step),\
             patch.object(native,'evaluate',side_effect=evaluate):
            state=native.train100(lambda pos:pos,labels,lambda s,e:None,validation,
                np.repeat([3,4,5,6],[1200,1200,1200,1100]),np.full(1000,2),
                all_steps.append,epoch)
        self.assertEqual((state['epoch'],state['updates']),(100,23500))
        self.assertEqual(len(all_steps),23500);self.assertEqual(len(epochs),100)
        self.assertEqual(model.training_calls,100)
        self.assertGreater(epochs[1]['validation_macro'],epochs[-1]['validation_macro'])
        self.assertEqual(int(state['model']['counter']),23500)

    def test_view_layout_is_exact_without_affine_or_random_augmentation(self):
        np=native.np
        raw=np.arange(2*3*90*189,dtype=np.float32).reshape(2,3,90,189)
        maps=raw.transpose(0,2,1,3).reshape(2,90,567)
        np.testing.assert_array_equal(engine.raw(maps,np.array([1,0])),raw[[1,0]])
        self.assertTrue(engine.raw(maps,np.array([1])).flags.c_contiguous)

if __name__=='__main__':unittest.main(verbosity=2)
