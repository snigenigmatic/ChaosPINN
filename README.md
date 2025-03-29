# ChaosPINN

## 🌊 Physics-Informed Neural Networks for the Kuramoto–Sivashinsky Equation

**ChaosPINN** is a SciML-powered project that leverages **Physics-Informed Neural Networks (PINNs)** to solve and visualize the **Kuramoto–Sivashinsky (KS) equation**, a nonlinear PDE that models chaotic wave propagation in fluid dynamics, plasma, and reaction-diffusion systems.

---

## 🚀 Problem Statement & Impact
### **Understanding and Predicting Chaotic Systems**
Traditional numerical solvers struggle with chaotic systems due to computational inefficiency and sensitivity to initial conditions. This project demonstrates how **PINNs can efficiently learn and predict chaotic wave evolution**, providing scalable and interpretable solutions for real-world applications like:
- **Turbulence modeling** in fluid mechanics 🌪️
- **Pattern formation** in chemical and biological systems 🧪
- **Plasma instabilities** in fusion reactors ⚡

---

## 🧠 Suggested Solution
We train a **PINN using PyTorch** to learn the underlying physics of the KS equation by minimizing both data loss and physics-based constraints. Our model directly enforces the governing PDE as a loss term, leading to better generalization and robustness.

### **Kuramoto–Sivashinsky Equation:**
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} = 0
$$

This equation exhibits chaotic behavior, making it an ideal test case for evaluating PINNs against traditional solvers.

---

## 🏠 Implementation Stack
- **Deep Learning Framework**: PyTorch ⚡
- **PINN Training**: Autograd-based physics loss functions 📉
- **Numerical Baseline**: Finite Difference Method (FDM) for comparison 🧪
- **Visualization**: Matplotlib & Plotly for 2D/3D wave evolution 🌊

---

## 🎥 Visualizations
Our model produces **real-time chaotic wave animations** and comparative plots:
- **Heatmap evolution of u(x,t)**
- **3D surface plots of chaotic wave patterns**
- **PINN predictions vs. numerical solutions**

![](demo.gif)  
(*Demo of PINN solving the KS equation*)

---

## 🔧 How to Run
### **1️⃣ Clone the Repository**
```bash
$ git clone https://github.com/yourusername/ChaosPINN.git
$ cd ChaosPINN
```

### **2️⃣ Install Dependencies**
```bash
$ pip install -r requirements.txt
```

### **3️⃣ Train the PINN**
```bash
$ python train.py
```

### **4️⃣ Visualize Results**
```bash
$ python visualize.py
```

---

## 🏆 Why This Project Stands Out
💯 **Stunning Visuals** – Chaos in action, animated! 🎥  
💯 **Scalable & Efficient** – PINNs offer a data-efficient alternative to traditional solvers 🚀  
💯 **Multi-Domain Impact** – Can extend to other nonlinear chaotic systems 🔥  

---

## 🌟 Authors
- **Aditya** - [LinkedIn](https://www.linkedin.com/in/aditya-sharma-pes) | [GitHub](https://github.com/Sharma-Aditya7) - ML & PINN Development 🧠
- **Anagha** - [LinkedIn](https://www.linkedin.com/in/anagha-rao-132b82287) | [GitHub](https://github.com/Anagha-Rao-53) - PINN Research & Optimization ⚙️
- **Kaustubh** - [LinkedIn](https://www.linkedin.com/in/c-kaustubh-413b77279) | [GitHub](https://github.com/snigenigmatic) - Web Integration & UI/UX 🎨
- **Purandar** - [LinkedIn](https://www.linkedin.com/in/purandar-puneet-918b92192) | [GitHub](https://github.com/PP-695) - Data Processing & Visualization 📊

---

## 👥 Contact & Contributions
We welcome contributions! Feel free to fork this repo, submit PRs, or reach out. 🚀
