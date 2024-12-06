# ConnectTheDots2
The goal of this project is to produce an ensemble model that accurately quantifies puncta formation in fluorescent images over a large dynamic range, these puncta are nucleation sites of DNA amplification. Previous work in the Posner Research group has shown that puncta counts correlate with the concentration of initial DNA concentration.

## Motivation
### Abstract
Nucleic Acid Tests (NATs) detect nucleic acid from disease and infection. Detection is based on amplifying low levels of DNA/RNA allowing for detection of a single strand of DNA/RNA. The gold standard for quantitative nucleic acid testing is quantitative polymerase chain reaction (qPCR). However, qPCR is:
* slow
* expensive 
* fragile 

Isothermal DNA amplification technologies, like recombinase polymerase amplification (RPA) have been put forth that are faster, cheaper, and more robust than qPCR. Yet isothermal amplification technologies are limited in their diagnostic capabilities as they are qualitative. However, **Recent studies in the Posner Lab Group have shown that RPA, an isothermal NAT, can also be quantitative through a spot nucleation to initial copy correlation** [1]. Similar nucleation site analysis has been applied to other assays and targets that used ML to produce a quantification model which rivals our linear range [2]. Thus, we are interested in applying ML models to improve the linear range of our assay.
1.  Quantitative Isothermal Amplification on Paper Membranes using Amplification Nucleation Site Analysis
Benjamin P. Sullivan, Yu-Shan Chou, Andrew T. Bender, Coleman D. Martin, Zoe G. Kaputa, Hugh March, Minyung Song, Jonathan D. Posner
bioRxiv 2022.01.11.475898; doi: https://doi.org/10.1101/2022.01.11.475898 
2. Membrane-Based In-Gel Loop-Mediated Isothermal Amplification (mgLAMP) System for SARS-CoV-2 Quantification in Environmental Waters
Yanzhe Zhu, Xunyi Wu, Alan Gu, Leopold Dobelle, Clément A. Cid, Jing Li, and Michael R. Hoffmann
Environmental Science & Technology 2022 56 (2), 862-873
DOI: 10.1021/acs.est.1c04623

For more information please see [Further details in the wiki](https://github.com/MartC53/QIAML/wiki/Further-details)

## Methods
### Image Prepocessing 
Prior to image processing, raw images were opened as numerical arrays for computational analysis. A Gaussian blur with a 5×5 kernel was applied to each image to soften edges and mitigate darkshot noise. To reduce interpuncta fluorescence, a temporal moving average was implemented by subtracting the average of the previous three frames from the current frame. Background subtraction was performed by averaging the first 20 frames to establish a baseline, which was subtracted from all subsequent frames to normalize pixel intensities to zero and remove autofluorescence artifacts. Finally, a circular mask was applied to exclude edge artifacts by setting the pixel intensities along the image boundaries to zero.
### Feature extraction 
The following features were extracted from the processed images to characterize puncta dynamics and fluorescence behavior. The maximum number of puncta identified over the entire time course was recorded, along with the number of puncta detected at 7.5 minutes. The temporal dynamics of puncta coverage were assessed by calculating the area occupied by puncta over time and the bulk fluorescence intensity of the puncta over time. Additionally, the percentage of the image area occupied by puncta at 7.5 minutes was quantified. The average size of individual puncta was determined as the total puncta area divided by the number of puncta. To capture temporal changes, the maximum rate of fluorescence change over a 10-second interval and the maximum rate of area change over a 10-second interval were computed.
### Model training  