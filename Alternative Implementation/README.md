# Alternative Implementation

The original project, [Real-time Detailed Video Analysis of Fruit Flies](https://cs229.stanford.edu/proj2018/report/61.pdf), implements the following:

| Feature | Original Implementation | Our Implementation |
| --- | --- | --- |
| Fly detection and counting | Decision Tree | Gaussian Kernel SVM + segmentation |
| Sex determination | Logistic Regression | Not implemented |
| Orientation estimation | HOG + PCA + Logistic Regression | HOG + PCA + Kernel Ridge Regression |
| Male wing-angle estimation | HOG + PCA + Linear Regression | Not implemented |
| Inter-fly distance | Not implemented | Logistic Regression + Linear Regression |
| Relative position | Not implemented | PCA + K-Means clustering |

## Key Extensions

Our alternative implementation introduces two additional analytical capabilities:

* **Inter-fly distance:** Estimates the distance between the male and female flies using their annotated pivot points.
* **Relative-position analysis:** Groups observations into eight clusters based on the flies' relative spatial positions and orientations.

Our implementation also explores different machine-learning approaches for fly detection and orientation estimation, extending the analytical scope of the original project.