# 🧠 Experience Report – ResNet + YOLO Object Detection

## 🔍 Project Summary
This project involved building an object detection model by integrating a YOLO-style detection head with a ResNet backbone and training it on a COCO-format dataset. The objective was to understand object detection fundamentals and explore how AI tools can support development.

---

## 💡 Challenges Faced

- **Understanding YOLO Output Structure**: Getting the grid-cell-based structure and anchor box logic to align with the COCO format was tricky.
- **Loss Function Implementation**: YOLO-style loss calculation is complex due to objectness, class, and bounding box regression components.
- **Data Loader for COCO-YOLO Format**: Custom parsing of label files in YOLO format took time to get right with resizing and matching boxes.

---

## 🤖 Use of AI Tools

- **ChatGPT**: Used to guide implementation of the YOLO head on ResNet, clarify how to reshape final layers, and understand NMS and anchor box logic.
- **GitHub Copilot**: Assisted in writing PyTorch training and evaluation boilerplate.
- **Stack Overflow + Docs**: Referenced solutions for Dataset class handling, loss debugging, and mAP evaluation.

---

## 📘 What I Learned

- Practical working of CNN backbones like ResNet and how feature maps can be extended with custom detection heads.
- The importance of data preprocessing and the impact of correct label formatting.
- Real-world object detection evaluation metrics like mAP, precision, and recall.

---

## 😲 What Surprised Me

- The sheer complexity of writing a YOLO model from scratch – especially loss functions and post-processing (like NMS).
- How easily AI tools accelerated my progress — but also how easy it is to misuse them without proper understanding.

---

## ⚖️ Coding Myself vs. AI Assistance

- **Balanced Approach**: I used AI tools to unblock me when stuck, but I wrote most of the logic after understanding what was required.
- **Code Ownership**: I made sure I understood every line even if it was suggested by AI, to ensure the learning was deep and authentic.

---

## 🧠 Suggestions for Improvement

- Provide a baseline or working YOLO head as a reference – it will save time and let us focus on integration and debugging.
- Introduce students to simple detection frameworks before jumping into custom builds.

---

✅ Overall, this project significantly boosted my understanding of computer vision and how to combine traditional deep learning with AI-assisted development.
