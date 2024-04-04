# DUENformer
The repository implements the DUENformer method through PyTorch.
Due to light scattering in underwater environments, underwater images often suffer from color distortion and low contrast, which can hinder the application of advanced computer vision tasks. To address the issue of serious color distortion and uneven degradation in underwater images, this paper proposes DUENformer, a dual encoder method for enhancing underwater images based on Transformer. The Axis Transformer Block (ATB) and Channel Transformer Block (CTB) are introduced to correct the colors of underwater images based on their specific characteristics. Additionally, the Spatial Interaction Module (SIM) and Channel Interaction Module (CIM) are proposed to incorporate local information into the Transformer block for effective fusion of local and global features. Meanwhile, in order to extract the features of objects in the underwater environment more flexibly, we introduce the Deformable Convolution (DeformConv) to capture the complex features of the underwater scene. The experimental results demonstrate that the DUENformer proposed in this paper outperforms nine classical methods. The qualitative results indicate that our method effectively corrects color bias problems and removes haze phenomena in underwater images, while also improving image contrast to align with human visual perception. Furthermore, quantitative evaluations show that DUENformer surpasses other methods proposed in this paper in terms of peak signal-to-noise ratio, structural similarity, and mean square error.
