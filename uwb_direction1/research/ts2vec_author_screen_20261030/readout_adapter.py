"""Author CLI RBF-SVM protocol, with job parallelism changed5->1 only.

Keeps the complete ten-value C grid including infinity, original five-fold
training-only CV, unscaled features and refit rule. No label/data loading here.
The optional old linear helper is not the CLI protocol and is not executed.
"""
import ast
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from author_adapter import BASE
def definition(parallel_only_adapter=True,grid_class=GridSearchCV):
 tree=ast.parse((BASE/'vendor/ts2vec/tasks/_eval_protocols.py').read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_svm');changed=0
 if parallel_only_adapter:
  for n in ast.walk(function):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='GridSearchCV':
    for kw in n.keywords:
     if kw.arg=='n_jobs': assert isinstance(kw.value,ast.Constant) and kw.value.value==5;kw.value.value=1;changed+=1
  assert changed==1
 ns=dict(np=np,SVC=SVC,GridSearchCV=grid_class,train_test_split=train_test_split)
 exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_author_SVM_protocol','exec'),ns);return ns['fit_svm']
def fit_svm(features,labels):return definition()(features,labels)
