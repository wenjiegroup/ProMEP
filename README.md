# ProMEP
Zero-shot prediction of mutation effects on protein function with multimodal deep representation learning
<<<<<<< HEAD

## Quick Start 
As a prerequisite, you must have  [SE(3)-Transformers](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer) installed to use this repository.

Dependences
```python
conda install --yes --file requirements.txt
```

## Usage 
Generate per-residue representations
```python
python inference.py --task ec --outfile embeddings.h5
```

Calculate log-ratio heuristic under the constraints of both sequence and structure

```python
python inference_dms.py --task ec --outfile fitness_prediction.h5
```

Zero-shot prediction of mutation effects
```python
python predict_mutation_effects.py testdata/fitness_prediction.h5 
```
=======
>>>>>>> origin/main
