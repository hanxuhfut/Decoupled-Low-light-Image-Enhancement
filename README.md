# Decoupled Low-light Image Enhancement

Shijie Hao<sup>1,2*</sup>, Xu Han<sup>1,2</sup>, Yanrong Guo<sup>1,2</sup> & Meng Wang<sup>1,2</sup>

<sup>1</sup>Key Laboratory of Knowledge Engineering with Big Data (Hefei University of Technology), Ministry of Education, Hefei 230009, China

<sup>2</sup>School of Computer Science and Information Engineering, Hefei University of Technology, Hefei 230009, China

---
TensorFlow implementation of the algorithm in the paper "**Decoupled Low-light Image Enhancement**".
tf1.8 & py3.6

**This paper has been accepted in the ACM Transactions on Multimedia Computing, Communications, and Applications**

## 1 Abstract
The visual quality of photographs taken under imperfect lightness conditions can be degenerated by multiple factors, e.g., low lightness, imaging noise, color distortion and so on. Current low-light image enhancement models focus on the improvement of low lightness only, or simply deal with all the degeneration factors as a whole, therefore leading to a sub-optimal performance. In this paper, we propose to decouple the enhancement model into two sequential stages. The first stage focuses on improving the scene visibility based on a pixel-wise non-linear mapping. The second stage focuses on improving the appearance fidelity by suppressing the rest degeneration factors. The decoupled model facilitates the enhancement in two aspects. On the one hand, the whole low-light enhancement can be divided into two easier subtasks. The first one only aims to enhance the visibility. It also helps to bridge the large intensity gap between the low-light and normal-light images. In this way, the second subtask can be shaped as the local appearance adjustment. On the other hand, since the parameter matrix learned from the first stage is aware of the lightness distribution and the scene structure, it can be incorporated into the second stage as the complementary information. In the experiments, our model demonstrates the state-of-the-art performance in both qualitative and quantitative comparisons, compared with other low-light image enhancement models. In addition, the ablation studies also validate the effectiveness of our model in multiple aspects, such as model structure and loss function.

## 2 Demo
- **Image 1**
![image1](/Demo/fig1.png)
- **Image 2**
![image2](/Demo/fig2.png)
- **Image 3**
![image3](/Demo/fig3.png)
- **Image 4**
![image4](/Demo/fig4.png)
- **Image 5**
![image5](/Demo/fig5.png)

Please consider to cite this paper if you find this code helpful for your research：

```

```
