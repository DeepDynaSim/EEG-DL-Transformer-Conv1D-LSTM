# EEG Deep Learning: Conv1D, LSTM and Transformers

**DeepDynaSim collection | Signal processing and machine learning | Research notebook**

This repository contains a TensorFlow/Keras EEG classification example and two explanatory PDFs. The implemented model concatenates a Conv1D-to-LSTM branch with a pooled attention branch, then applies a dense softmax classifier.

## Resources

- [Notebook](EEG-DL%20Transformer%2BConv1D%2BLSTM.ipynb)
- [Technical report](Deep%20Learning%20in%20EEG%20Signal%20Processing_Advanced%20Methods_Technical%20Report.pdf)
- [Code explanation](Code%20Explanation.pdf)

## Environment and input data

The initial cells mount Google Colab Drive and read `/content/gdrive/MyDrive/inputdata.mat` and `labeldata.mat`. The required fields are `inputdata` and `labeldata`; these files are not included. For local Jupyter use, replace the Drive-mount cell and set local paths.

```sh
python -m pip install notebook tensorflow numpy scipy scikit-learn matplotlib psutil
python -m notebook
```

Dependency versions are not locked. `psutil` is used by an initial system-information cell; TensorFlow/Keras, NumPy, SciPy, scikit-learn and Matplotlib support the numerical workflow.

The model expects `X` to have shape `(samples, time_steps, 1)`. A two-dimensional input must contain samples in rows and time steps in columns. The explicit train/validation/test reshapes now write back to the three named arrays; rebinding a temporary loop variable does not reshape those arrays.

Labels may be a one-dimensional vector, a row/column vector, or a valid one-hot matrix. Vector labels are mapped to a fixed sorted `class_values` array before one-hot encoding. Preserve that mapping when interpreting predicted class indices. Input and label sample counts must agree.

## Executed protocol and limitations

The notebook uses a row-wise random split with seed 42: 20 percent test, then 25 percent of the remainder for validation, leaving 60/20/20 percent training/validation/test. It does not implement subject- or session-grouped splitting. Adapt the split to the dataset's identifiers before making subject-independent claims.

Training uses Adam, categorical cross-entropy, batch size 64 and up to 30 epochs, with early stopping, checkpointing and learning-rate reduction. `best_hybrid_model.h5` is written in the working folder. Data preprocessing and model defaults must be reviewed for the actual recording arrangement and installed Keras version.

The original architecture is retained. In particular, the attention branch's fixed dense projection and single-channel normalization need architectural evaluation; this repository does not supply a controlled model-comparison benchmark or establish that the attention branch improves classification. The input-shape and label fixes do not constitute new performance evidence. Saved notebook outputs have been cleared after those fixes; training was not repeated on unavailable source data.

## Conceptual pipeline figure

The supplied figure depicts a broader acquisition-to-interface concept. Its 19-channel/125-Hz acquisition details, ICA preprocessing, cloud service and web interface are not implemented or verified by this notebook. Do not infer those recording settings for the supplied MAT files.

![Conceptual EEG acquisition, classification and interface pipeline; implementation scope is described above](https://github.com/DeepDynaSim/EEG-DL-Transformer-Conv1D-LSTM/assets/35817872/402fdae4-1ca3-45dc-a070-0834a70c8dd0)

The example does not establish clinical diagnostic validation. Report the source data, class mapping, recording layout, group separation, software versions and evaluation protocol with any new result.

## Related work and collaboration

Explore related repositories: [atr ftir 2dfft robustness](https://github.com/DeepDynaSim/atr-ftir-2dfft-robustness) · [skin lesion gnn stability](https://github.com/DeepDynaSim/skin-lesion-gnn-stability).

[DeepDynaSim](https://github.com/DeepDynaSim) is an engineering R&D and consulting company in İzmir, Türkiye, focused on physics-guided AI, simulation and control. For engineering collaboration: [support@deepdynasim.com](mailto:support@deepdynasim.com). For repository questions, use [Issues](https://github.com/DeepDynaSim/EEG-DL-Transformer-Conv1D-LSTM/issues).

Reuse follows the existing [LICENSE](LICENSE); academic authorship and third-party attribution remain with their respective creators.
