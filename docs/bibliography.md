# Bibliography / Literature Review

## Facial Emotion Recognition

1. Goodfellow, I., Erhan, D., Carrier, P.L., Courville, A., Mirza, M., Hamner, B., ... & Bengio, Y. (2013).
   *Challenges in Representation Learning: A report on three machine learning contests*.
   In International Conference on Neural Information Processing (pp. 117–124). Springer, Berlin, Heidelberg.
   https://arxiv.org/abs/1307.0414

   The original publication introducing the FER-2013 dataset used in this project. Provides background on the
   challenge of facial emotion recognition and the dataset's construction methodology.

2. Ammar, S., Bouwmans, T., & Neji, M. (2022).
   *Face Identification Using Data Augmentation Based on the Combination of DCGANs and Basic Manipulations*.
   Information, 13(8), 370.
   https://doi.org/10.3390/info13080370

   Used as reference for data augmentation techniques in facial emotion recognition, and as inspiration
   for the DCGAN-based synthetic face generation component of this project.

3. Kim, J.-H., Kim, N., & Won, C. S. (2022).
   *Facial Expression Recognition with Swin Transformer*.
   https://arxiv.org/abs/2203.13472

   Used as inspiration for modern FER model architectures. Informed the decision to implement a custom
   Vision Transformer (ViT) as part of the model comparison experiments.

4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.C. (2018).
   *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.
   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520.
   https://arxiv.org/abs/1801.04381

   Core reference for the MobileNetV2 architecture used as the final production model via transfer learning.
   Explains the depthwise separable convolution design that makes MobileNetV2 efficient for mobile and
   resource-constrained applications.

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020).
   *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*.
   International Conference on Learning Representations (ICLR 2021).
   https://arxiv.org/abs/2010.11929

   The original Vision Transformer (ViT) paper. Motivated the implementation of a custom mini-ViT
   architecture as part of the model exploration, demonstrating the applicability of the Transformer
   self-attention mechanism to image classification tasks.

## Emotion-Based Music Recommendation

6. Grekow, J.
   *Music Emotion Maps in Arousal-Valence Space*.
   Bialystok University of Technology.
   https://www.researchgate.net/publication/307909024_Music_Emotion_Maps_in_Arousal-Valence_Space

   Used for mapping detected emotions into musical mood dimensions (tempo, key, energy) using the
   Russell circumplex model. Directly informed the design of the association rule mapping table.

7. Athavle, M., Mudale, D., Shrivastav, U., & Gupta, M. (2021).
   *Music Recommendation Based on Face Emotion Recognition*.
   Journal of Informatics Electrical and Electronics Engineering, 2(2), 1–11.
   https://www.researchgate.net/publication/354855186_Music_Recommendation_Based_on_Face_Emotion_Recognition

   Core reference for the overall system architecture combining FER and music recommendation.
   Validated the pipeline approach: detect face → classify emotion → recommend music.

8. Agrawal, R., & Srikant, R. (1994).
   *Fast Algorithms for Mining Association Rules*.
   In Proceedings of the 20th International Conference on Very Large Data Bases (VLDB), 487–499.

   Original paper introducing the Apriori algorithm used in the association_rules.py module
   for discovering emotion → music attribute relationships. Foundational reference for the
   rule mining methodology applied in this project.
