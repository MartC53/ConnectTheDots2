ConnectTheDots2: Functional Specification

Background

Without a cure, antiretroviral therapy (ART) is the primary tool for treating individuals living with HIV (Drain et Al. 2020). Monitoring HIV viral load is an important element of care for individuals on ART, as virally suppressed patients cannot transmit HIV to others. As a result, combating the HIV epidemic requires timely and accurate viral load testing.
Despite the importance of viral load testing, existing testing technologies present implementation challenges, particularly in low- and middle-income countries (Grieg et Al, 2020). For example, the current gold standard for testing—quantitative polymerase chain reaction (qPCR)—requires expensive lab equipment, highly specialized technicians, and large amounts of time. 
Isothermal DNA amplification technologies, like recombinase polymerase amplification (RPA) have been put forth that are faster, cheaper, and more robust than qPCR. However, these technologies currently provide qualitative—rather then quantitative viral load counts—of HIV infection. As a result, researchers are working to adapt RPA testing methods to produce accurate viral load counts.
Recent studies in the Posner Lab Group have shown that RPA can also be quantitative through a spot nucleation to initial copy correlation. Currently, the technology can measure viral load at lower levels. Predicting viral load using RPA at higher viral loads remains a challenge.
The goal of this project is to produce an ensemble model that accurately quantifies puncta formation in fluorescent images over a large dynamic range in RPA samples, these puncta are nucleation sites of DNA amplification. Building on past work, our approach improves the predictive accuracy of puncta counts using labeled data extracted from RPA samples.
User profile. Who uses the system. What they know about the domain and computing (e.g., can browse the web, can program in Python)

Data sources

Our dataset includes features extracted from images of RPA samples, along with puncta counts for use in training and testing. Features were extracted by the main project team using existing workflows, with the goal of training a machine learning model which can improve on existing prediction methods without investing in new feature extraction processes. 
Prior to image processing, raw images were opened as numerical arrays for computational analysis. A Gaussian blur with a 5×5 kernel was applied to each image to soften edges and mitigate darkshot noise. To reduce interpuncta fluorescence, a temporal moving average was implemented by subtracting the average of the previous three frames from the current frame. Background subtraction was performed by averaging the first 20 frames to establish a baseline, which was subtracted from all subsequent frames to normalize pixel intensities to zero and remove autofluorescence artifacts. Finally, a circular mask was applied to exclude edge artifacts by setting the pixel intensities along the image boundaries to zero.

The following features were extracted from the processed images to characterize puncta dynamics and fluorescence behavior. The maximum number of puncta identified over the entire time course was recorded, along with the number of puncta detected at 7.5 minutes. The temporal dynamics of puncta coverage were assessed by calculating the area occupied by puncta over time and the bulk fluorescence intensity of the puncta over time. Additionally, the percentage of the image area occupied by puncta at 7.5 minutes was quantified. The average size of individual puncta was determined as the total puncta area divided by the number of puncta. To capture temporal changes, the maximum rate of fluorescence change over a 10-second interval and the maximum rate of area change over a 10-second interval were computed.

Use Cases

Use case 1: Main Project team
Primary users of our model will be Cole Martin and collaborators, which are development a phone application to predict viral load using RPA samples. This group has a high level of subject matter expertise, developed the underlying test chemistry, and will engage directly with our models. The main project team will need to (1) incorporate our model into their existing phone application and (2) update our model as new data and prediction algorithms become available.

Use case 2: Clinician testing phone application 

As the phone app supported by our models matures, clinicians testing the application in the field will use our model to predict viral loads using samples taken from patients. Clinicians have a high level of experience working with people living with HIV and need quick and accurate viral load readings to guide treatment. However, clinicians have less experience with the underlying technology. Clinicians will interact with our model only when reading samples into the phone application and obtaining viral load readings, with the goal of monitoring and treating patients. Therefore, it is essential that our model balances accuracy and speed.
