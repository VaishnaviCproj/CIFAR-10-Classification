# CIFAR-10 Classification Using Machine Learning and Deep Learning Models

### Project Overview
This project explored image classification on the CIFAR-10 dataset using machine learning and deep learning models — Random Forest, CNN, and ResNet. The goal was to compare the performance of these models to understand their strengths and limitations for image recognition tasks. While Random Forest served as a baseline, CNNs captured spatial hierarchies, and ResNet leveraged residual connections to improve training efficiency.  

---

### Skills  
- **Programming & Frameworks**: Python, PyTorch, scikit-learn  
- **Data Engineering & Processing**: Apache Airflow, PCA, t-SNE, data normalization, auto-augmentation  
- **Model Development**: Random Forest, Convolutional Neural Networks (CNN), ResNet-18  
- **Model Optimization**: Hyperparameter tuning, learning rate scheduling, batch normalization, dropout regularization  
- **Visualization & Analysis**: Matplotlib, Seaborn, 3D PCA visualizations, 2D t-SNE clustering  

---

### Key Contributions 
- **Data Preprocessing**: Prepared the CIFAR-10 dataset with normalization and auto-augmentation, dividing it into training, validation, and test sets. Applied PCA to reduce dimensionality and streamline training.  
- **Model Implementation**:  
  - **Random Forest**: Tuned hyperparameters like tree depth, feature selection, and leaf samples, achieving a baseline accuracy of **44.97%**.  
  - **CNN**: Designed a **3-layer CNN** with batch normalization, dropout, and max pooling, carefully adjusting hyperparameters like learning rate and batch size, reaching **81.1%** accuracy.  
  - **ResNet**: Implemented **ResNet-18** with modified convolution layers and residual connections, experimenting with optimizers and activation functions to achieve the highest accuracy of **83.6%**.  
- **Exploratory Data Analysis**: Visualized class distributions, applied PCA for **3D projections**, and used **t-SNE for 2D clustering** to assess class separability.  

---

### Results and Impact
- Demonstrated the performance gap between **traditional machine learning and deep learning models**, with ResNet outperforming Random Forest by nearly **39% in accuracy**.  
- Showed that **deep learning models**, through convolutional and residual layers, **better capture spatial and hierarchical features**, making them more suitable for complex image classification tasks.  
- Highlighted the **trade-offs between model complexity and accuracy**, providing insights for **real-world applications** like object recognition, e-commerce, and autonomous systems.  

---

### Learnings and Takeaways  
- Gained hands-on experience in **model selection, hyperparameter tuning, and the practical challenges of training deep networks**.  
- Understood the significance of **residual connections in mitigating the vanishing gradient problem**, enabling deeper architectures to converge effectively.  
- Learned how **data preprocessing, dimensionality reduction, and visualization techniques** can guide model development and improve interpretability.  
