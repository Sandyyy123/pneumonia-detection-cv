# Additional References - RSNA Pneumonia Detection (Project #15)

Independent literature scan (2024-2026), verified live against Europe PMC and CrossRef. Volume/issue/pages omitted per project rules. Format: Authors. Title. Journal. Year. DOI.

## State-of-the-art callout: gaps in current `reports/references.md`

The existing reference list is methodologically strong on classical detectors (RetinaNet/FPN/Faster R-CNN/YOLOv1-v2/SSD/Mask R-CNN), CNN backbones (ResNet/DenseNet/MobileNetV2), and the canonical CXR ecosystem (ChestX-Ray8, CheXNeXt, RSNA descriptor, Zech, Majkowska). Five concrete gaps the project SHOULD cite before Phase 2:

1. **Foundation models for CXR.** Yao 2025 (EVA-X) and Yang 2025 (CXR-FM with global+local representations) report that self-supervised CXR pretraining cuts the labelled-data budget for downstream detection by 2-3x. The manuscript already flags this in Section 5.7 "Future work" but cites no 2024-2026 anchor.
2. **YOLOv8 (and v11/v12) primary citations.** The advanced model is YOLOv8 but the existing references only cite YOLOv1 (Redmon 2016) and YOLO9000. A current-generation YOLO citation (Sahingoz 2026 systematic review v1-v12; Mao 2026 YOLOv12 lung nodule) is needed.
3. **Transformer / DETR detectors as a comparator family.** The Methods discusses single-stage vs two-stage but does not mention DETR-style set-prediction detectors. Wang 2026 (CNN-to-transformer evolution) and Kumar 2025 (DETR for medical ultrasound) are relevant.
4. **Bias mitigation, not just bias diagnosis.** Existing list cites Zech 2018 for bias detection but nothing for mitigation. Mottez 2026 (PSB; from-detection-to-mitigation on CXR) and Rehman 2026 (diffusion-synthesised CXRs improve fairness) close that gap.
5. **Pneumonia-specific 2024-2026 work using exactly the RSNA pipeline shape.** Xie 2025 (YOLO-based RSNA pneumonia diagnosis), Slimi 2025 (attention-guided trustworthy detection), Hegazy 2026 (EfficientNetB0 + CLAHE) directly map to the baseline/advanced choices and let the manuscript triangulate against current numbers.

The current Section 5.7 also flags self-supervised pretraining and external validation; both have strong 2024-2026 anchors below that should be promoted into the manuscript citations.

---

## Architectures and detectors (2024-2026)

1. Wang Z, Chen Y, Gu Y, Liu J, Zhu X, He M. The evolution of object detection from CNNs to transformers and multi-modal fusion. Scientific Reports. 2026. DOI:10.1038/s41598-026-37052-6
2. Sahingoz OK, Karatas Baydogmus G, Kugu E. A Systematic Literature Review of You Only Look Once Architectures (v1-v12) in Healthcare Systems. Diagnostics. 2026. DOI:10.3390/diagnostics16060935
3. Mao M, Hong C, Zhang Y, Huang H, Chu J, Fu L. Research on lung nodule detection in X-ray plain films based on improved YOLOv12 model. Scientific Reports. 2026. DOI:10.1038/s41598-026-47670-9
4. Kumar D, Mehta MA, Müller H, Holzinger A. Transformer-powered precision: A DETR-based approach for robust detection in medical ultrasound with cholelithiasis as a case study. Computational and Structural Biotechnology Journal. 2025. DOI:10.1016/j.csbj.2025.10.037
5. Shehzadi T, Ifza I, Liwicki M, Stricker D, Afzal MZ. Semi-Supervised Object Detection: A Survey on Progress from CNN to Transformer. Sensors. 2026. DOI:10.3390/s26010310
6. Qin X, Qian Q, Li X, Deng C, Wang W, Peng L, Lu H, Gong Y, Zhao J. ScaleMamba-YOLO: a multi-scale MambaYOLO for medical object detection. Scientific Reports. 2026. DOI:10.1038/s41598-026-37258-8

## Pneumonia and chest-radiograph deep learning (2024-2026)

7. Slimi H, Balti A, Abid S, Sayadi M. Trustworthy pneumonia detection in chest X-ray imaging through attention-guided deep learning. Scientific Reports. 2025. DOI:10.1038/s41598-025-23664-x
8. Xie Y, Zhu B, Jiang Y, Zhao B, Yu H. Diagnosis of pneumonia from chest X-ray images using YOLO deep learning. Frontiers in Neurorobotics. 2025. DOI:10.3389/fnbot.2025.1576438
9. Siddiqi R, Javaid S. Deep Learning for Pneumonia Detection in Chest X-ray Images: A Comprehensive Survey. Journal of Imaging. 2024. DOI:10.3390/jimaging10080176
10. Saranyaraj D, Shrinaath V, Nayak A, Vishal R. PneuNet: a lightweight convolutional neural network with multiscale feature fusion for automated pneumonia detection from chest X-rays. Frontiers in Medicine. 2025. DOI:10.3389/fmed.2025.1713587
11. Sebastian N, Ankayarkanni B. Enhanced ResNet-50 with Multi-Feature Fusion for Robust Detection of Pneumonia in Chest X-Ray Images. Diagnostics. 2025. DOI:10.3390/diagnostics15162041
12. Adam Essa ME. Diagnostic accuracy of AI in chest radiography for pneumonia and lung cancer: A meta-analysis. European Journal of Radiology Open. 2025. DOI:10.1016/j.ejro.2025.100701
13. Hegazy NY, Sawah MS. An optimized EfficientNetB0 framework with CLAHE-based preprocessing for accurate multi-class chest X-ray classification. Scientific Reports. 2026. DOI:10.1038/s41598-026-42492-1
14. Aldabayan YS. Pneumonia and pneumothorax detection: A multi-factor evaluation of chest X-rays. PLOS ONE. 2026. DOI:10.1371/journal.pone.0341060
15. Guan L, Zhang R, Zhao Y. YOLOv11-MFF: A multi-scale frequency-adaptive fusion network for enhanced CXR anomaly detection. PLOS ONE. 2025. DOI:10.1371/journal.pone.0334283
16. Reddy KD, Patil A. CXR-MultiTaskNet: a unified deep learning framework for joint disease localization and classification in chest radiographs. Scientific Reports. 2025. DOI:10.1038/s41598-025-16669-z
17. Ye Y, Zhou W. Artificial intelligence in the diagnosis and prognosis of pediatric bacterial pneumonia: current advances and challenges. Current Opinion in Pediatrics. 2026. DOI:10.1097/MOP.0000000000001545

## Foundation models, self-supervised pretraining, vision-language (2024-2026)

18. Yao J, Wang X, Song Y, Zhao H, Ma J, Chen Y, Liu W, Wang B. EVA-X: a foundation model for general chest x-ray analysis with self-supervised learning. NPJ Digital Medicine. 2025. DOI:10.1038/s41746-025-02032-z
19. Yang Z, Xu X, Zhang J, Wang G, Kalra MK, Yan P. Chest X-Ray Foundation Model With Global and Local Representations Integration. IEEE Transactions on Medical Imaging. 2025. DOI:10.1109/TMI.2025.3581907
20. Wang F, Yu L. Scaling Chest X-Ray Foundation Models From Mixed Supervisions for Dense Prediction. IEEE Transactions on Medical Imaging. 2025. DOI:10.1109/TMI.2025.3589928
21. Ahmad IS, Suleiman RB, Yu T, Lin R, Zhao C, Liao J, Wu B, Xie Y, Liang X, Wang H, Hu Z. Foundation models for X-ray interpretation: a narrative review of current techniques and future perspectives in diagnostic imaging. Quantitative Imaging in Medicine and Surgery. 2026. DOI:10.21037/qims-2025-1-2782
22. Ryu JS, Kang H, Chu Y, Yang S. Vision-language foundation models for medical imaging: a review of current practices and innovations. Biomedical Engineering Letters. 2025. DOI:10.1007/s13534-025-00484-6
23. Zambrano Chaves JM, Huang SC, Xu Y, Xu H, Usuyama N, Zhang S, et al. A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings. Nature Communications. 2025. DOI:10.1038/s41467-025-58344-x
24. Yoon T, Kang D. Integrating snapshot ensemble learning into masked autoencoders for efficient self-supervised pretraining in medical imaging. Scientific Reports. 2025. DOI:10.1038/s41598-025-15704-3
25. Hommyo M, Tsuji T, Kumagai S, Shiraishi K, Kotoku J. A Practical Study of Data Requirements for Self-Supervised Learning in Medical Image Analysis. Cureus. 2025. DOI:10.7759/cureus.98049

## Generalisation, dataset bias, and fairness (2024-2026)

26. Mottez C, Fay L, Varma M, Ostmeier S, Langlotz C. From Detection to Mitigation: Addressing Bias in Deep Learning Models for Chest X-Ray Diagnosis. Pacific Symposium on Biocomputing. 2026. DOI:10.1142/9789819824755_0039
27. Rehman D, Chen HY, Lee CC, Vilor-Tejedor N, Upadhyay S, Kuo PC. Diffusion-synthesized Chest X-rays improve fairness and diagnostic performance. PLOS Digital Health. 2026. DOI:10.1371/journal.pdig.0001277
28. Singthongchai J, Wangkhamhan T. Adaptive Normalization Enhances the Generalization of Deep Learning Model in Chest X-Ray Classification. Journal of Imaging. 2025. DOI:10.3390/jimaging12010014
29. Gao Y, Hao J, Zhou B. FairREAD: Re-fusing demographic attributes after disentanglement for fair medical image classification. Medical Image Analysis. 2026. DOI:10.1016/j.media.2025.103858
30. Pathak AK, Gupta M, Jain G. Unmasking the Clever Hans effect in AI models: shortcut learning, spurious correlations, and the path toward robust intelligence. Frontiers in Artificial Intelligence. 2025. DOI:10.3389/frai.2025.1692454
31. Mohammadi M, Vejdanihemmat M, Lotfinia M, Rusu M, Truhn D, Maier A, Tayebi Arasteh S. Differential privacy for medical deep learning: methods, tradeoffs, and deployment implications. NPJ Digital Medicine. 2026. DOI:10.1038/s41746-025-02280-z

## Datasets, benchmarks, and surveys (2024-2026)

32. Khan FK, Tahir WB, Lee MS, Kim JY, Byon SS, Pi SW, Lee BD. Leveraging Large-Scale Public Data for Artificial Intelligence-Driven Chest X-Ray Analysis and Diagnosis. Diagnostics. 2026. DOI:10.3390/diagnostics16010146
33. Montalvo-Lezama B, Fuentes-Pineda G. MetaChest: generalized few-shot learning of pathologies from chest X-rays. Visual Computing for Industry, Biomedicine, and Art. 2026. DOI:10.1186/s42492-026-00214-4
34. Almughamisi N, Abosamra G, Albar A, Saleh M. Hybrid ConvNeXtV2-ViT Architecture with Ontology-Driven Explainability and Out-of-Distribution Awareness for Transparent Chest X-Ray Diagnosis. Diagnostics. 2026. DOI:10.3390/diagnostics16020294
35. Lokunde V, Sundar K, Khokhar A, Tyagi B, R NP, B M. Class-attention pooling and token sparsity based vision transformers for chest X-ray interpretation. Scientific Reports. 2026. DOI:10.1038/s41598-026-37109-6
36. Soriano JB, Lumbreras S. The rise of artificial intelligence in respiratory primary care and pulmonology: a scoping review. NPJ Primary Care Respiratory Medicine. 2026. DOI:10.1038/s41533-026-00487-5

## Federated, privacy, and deployment (2024-2026)

37. Serrano ALM, Rodrigues GAP, Bispo GD, Gonçalves VP, Filho GPR, Peixoto MGM, Bonacin R, Meneguette RI. FedIHRAS: A Privacy-Preserving Federated Learning Framework for Multi-Institutional Collaborative Radiological Analysis with Integrated Explainability and Automated Clinical Reporting. Biomedicines. 2026. DOI:10.3390/biomedicines14030713
38. P K, P S, Ar NK. Proximal guided hybrid federated learning approach with parameter efficient adaptive intelligence for pneumonia diagnosis. Scientific Reports. 2025. DOI:10.1038/s41598-025-32286-2
39. Rajaraman S, Marini N, Liang Z, Xue Z, Antani S. Multimodal Learning with Privileged Report Supervision for Generalizable Tuberculosis Detection on Chest Radiographs. Journal of Medical Systems. 2026. DOI:10.1007/s10916-026-02368-3
40. Chiang LF. An Interpretable Chest X-ray Classification Framework Using Prototype Memory and Counterfactual Consistency. Cureus. 2026. DOI:10.7759/cureus.103134
