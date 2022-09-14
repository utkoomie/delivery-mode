## Notebooks and Companion Code for Investigation Into the Optimal Mode of Delivery in Pregnancy Using National Vital Statistics Data
Analysis is contained within 3 notebooks:

* [loadBirthData](loadBirthData.ipynb)
* [deliveryMode](deliveryMode.ipynb)
* [compareML_models](compareML_models.ipynb)

A Dockerfile is provided which can be used to generate a container with all necessary software to run these notebooks. Prior to use, datafiles containing all births from 2005-2017 must be downloaded from the CDC. These files are publicly available for download at https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm#Births. Note that the uncompressed size of these files totals approximately 50GB.
