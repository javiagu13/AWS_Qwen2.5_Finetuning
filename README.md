# Train VLMs in Large AWS Clusters

This repository was created to address the lack of open-source tools for training Visual Language Models (VLMs) in large AWS clusters. With dedicated support for the **Qwen 2.5 VL** model, this solution tackles the unique challenges of training multimodal models in cloud environments.

---

## 🚀 Features

- **AWS Cluster Compatibility**: Designed for AWS `p4d` and `p4de` clusters, each equipped with 8 A100 GPUs.
  - **p4d**: Each GPU provides **40GB** memory.
  - **p4de**: Each GPU provides **80GB** memory.
- **Supported Model**: Specifically built for the **Qwen 2.5 VL** model.
- **Performance**:
  - Average training time:
    - **p4d**: ~1 hour (with gradient offloading to CPU due to memory constraints).
    - **p4de**: ~30 minutes (with full CUDA-based training).
  - Memory usage:
    - **p4d**: ~37GB per GPU.
    - **p4de**: ~47GB per GPU (higher due to reduced offloading needs).

---

## 🛠️ Challenges Addressed

Training VLMs on AWS clusters comes with unique technical challenges. This repository tackles key obstacles, including:

1. **Containerized Training**:

   - Since training jobs are executed inside containers, the code must be tailored to work seamlessly within this environment.

2. **Parallelization Issues**:
   - Enabling tools like DeepSpeed to handle parallelization (e.g., model, gradient, and optimizer sharding) was a significant hurdle.
   - While 8 A100 GPUs may seem sufficient, training in mixed precision (BF16) poses challenges due to GPU memory limits. Native PyTorch techniques like `DataParallel` and `ModelParallel` often crash due to excessive GPU consumption, even in large clusters.

---

## 🧠 Our Approach

We leverage **DeepSpeed** to handle the complex requirements of parallelization and optimization, ensuring efficient utilization of GPU memory and computational power. By striking the right balance between speed and resource use, our method proves to be more economical than using slower, cheaper clusters.

### Key Components

This repository includes an essential notebook to streamline the workflow:

1. **AWS Training Notebook**: Configures and executes the training job in an AWS environment.

---

## 📊 Why This Repository?

Despite the high cost of AWS clusters, the speed of training achieved using this repository makes it more economical compared to prolonged use of cheaper clusters. By overcoming critical technical barriers, this repository provides a robust, reusable solution for researchers and engineers working with state-of-the-art VLMs.

---

## 🏆 Tested Models and Research Use Cases

This repository has been extensively tested and used for training the **Qwen 2.5 VL** model across various research projects. Its effectiveness and versatility make it an invaluable tool for anyone working with multimodal models.

---

## 🛡️ Disclaimer

To use this repository effectively, familiarity with AWS services (e.g., EC2, training jobs) and deep learning tools (e.g., PyTorch, DeepSpeed) is recommended.

## Use

The codes ending with CPPE-5 have been thoroughly tested and used for finetuning. It is recommended to use those two codes (Training and AWQ codes). The main idea behind this is to train on CPPE-5 dataset. If you want to use other dataset. create a dataset in huggingface and update the code.
