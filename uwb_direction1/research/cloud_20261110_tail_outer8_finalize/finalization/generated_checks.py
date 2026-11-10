"""Metric and fixed-gate edge cases only, no real arrays/truth access."""
from common import *
from metrics import classification_metrics
def main():
 a=classification_metrics(np.array([0,0,0,1]),np.array([0,0,0,0]));assert a['accuracy']==.75 and a['present_class_macro_recall']==.5 and a['class_recall'][2] is None
 b=classification_metrics(np.array([0,0,1,1]),np.array([0,2,1,1]));assert b['accuracy']==.75 and b['missing_classes']==list(range(2,12))
 rows=[]
 for h in range(1,9):
  y=np.zeros(1 if h==1 else 3,dtype=int);base=classification_metrics(y,np.zeros(len(y),dtype=int) if h==1 else np.ones(len(y),dtype=int));good=classification_metrics(y,np.zeros(len(y),dtype=int));rows.append(dict(held=h,native=base,tail=good,native_sourceBN=base))
 avg=aggregate(rows,'native');assert avg['equal_user_mean_accuracy']==.125 and avg['pooled_sample_accuracy']==1/22
 pair=paired(rows,'tail');assert pair['strict_accuracy_wins']==7 and pair['accuracy_ties']==1 and pair['worst_paired_accuracy_delta']==0
 cfg=read(ROOT/'PROTOCOL.json')['gate'];assert gate(pair,cfg)
 exact=dict(mean_accuracy_delta=.005,strict_accuracy_wins=5,worst_paired_accuracy_delta=-.02);assert gate(exact,cfg)
 for k,v in [('mean_accuracy_delta',.004),('strict_accuracy_wins',4),('worst_paired_accuracy_delta',-.021)]:assert not gate(dict(exact,**{k:v}),cfg)
 try:paired(rows[:-1],'tail')
 except AssertionError:pass
 else:raise AssertionError('Partial cohort accepted')
 write(HERE/'GENERATED_CHECKS.json',dict(state='pass',unbalanced_accuracy_vs_macro_distinct=True,missing_class_null=True,absent_class_prediction_counted_wrong=True,equal_user_vs_pooled_distinct=True,partial_cohort_blocked=True,full8_accuracy_gate_boundaries_pass=True,real_gesture_truth_values_read=False))
 print('Generated full8 metric/gate checks passed; real truth unread')
if __name__=='__main__':main()
