# [CVPR2025 Workshops] Domain Generalization through Attenuation of Domain-Specific Information

[reiji saito](https://zxwei.site), [kazuhiro hotta]() <br />
Meijo University (JAPAN)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-generalization-through-attenuation-of/domain-generalization-on-cityscapes-to-acdc)](https://paperswithcode.com/sota/domain-generalization-on-cityscapes-to-acdc?p=domain-generalization-through-attenuation-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-generalization-through-attenuation-of/domain-generalization-on-gta5-to-cityscapes)](https://paperswithcode.com/sota/domain-generalization-on-gta5-to-cityscapes?p=domain-generalization-through-attenuation-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-generalization-through-attenuation-of/domain-generalization-on-gta-to-avg)](https://paperswithcode.com/sota/domain-generalization-on-gta-to-avg?p=domain-generalization-through-attenuation-of)
<br />
Paper: https://openaccess.thecvf.com/content/CVPR2025W/DG-EBF/html/Saito_Domain_Generalization_through_Attenuation_of_Domain-Specific_Information_CVPRW_2025_paper.html

![ADSI Framework](docs/ADSI.png)

# ADSI when using DINOv2
|Setting |mIoU |Config|Log & Checkpoint|
|-|-|-|-|
|**GTAV $\rightarrow$ Cityscapes**|**67.75**|[config](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/ADSI.py)|[log](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/20250101_172427.log) & [checkpoint](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/ADSI_gta2avg.pth)
|**GTAV $\rightarrow$ BDD100K**|**61.38**|[config](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/ADSI.py)|[log](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/20250101_172427.log) & [checkpoint](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/ADSI_gta2avg.pth)
|**GTAV $\rightarrow$ Mapillary**|**67.59**|[config](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/ADSI.py)|[log](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/20250101_172427.log) & [checkpoint](https://github.com/ReijiSoftmaxSaito/ADSI/releases/download/v1.0/ADSI_gta2avg.pth)