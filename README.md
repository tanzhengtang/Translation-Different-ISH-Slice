# Translation-Different-ISH-Slice
By using the framework of CycleGAN and pix2pix to train the model with the large number of in situ hybridization(ISH) slices data of mouse brain in Allen Brain Altases, achieving that show different gene expression on one same tissue ISH section slice without another different tissue section.  

* The repo of cyclegan: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
* The link of allen brain altases: https://mouse.brain-map.org/ 
* The code framework is from https://github.com/miracleyoo/pytorch-lightning-template.git   
* The link of lightning package: https://lightning.ai/

The principle of in situ hybridization(ISH) determines its low-throughput nature, which means that one tissue corresponding to only one gene. Then the prerequisite of joint multi-gene analysis on different ISH tissue section is **one precise registration** of those tissue slice to **an common standard space**. After that implementation of a relatively accurate analysis between different genes could be started. In fact, an accurate cross-modal registration is somewhat difficult, and ISH section is usually accompanied by damage, deformation and contamination, which further increases the difficulty of registration.  

The development of modal conversion in medical imaging especially virtual staining of HE images is the main inspiration that sparked this project. Essentially, the virtual staining of HE image is using the local texture information in UV-style tissue slices to generate the corresponding staining of cell nuclei and cytoplasm, which depends on the prior distribution of stain infomation in the training data.  

etc...