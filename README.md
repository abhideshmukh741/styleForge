# StyleForge - Neural Style Transfer Web Application

**StyleForge** is a state-of-the-art web application that transforms your ordinary images into extraordinary works of art using **Neural Style Transfer (NST)**. Leveraging deep learning models, StyleForge separates the content of one image from the artistic style of another, merging them into a seamless, breathtaking new creation.

![StyleForge Demo](static/images/demo.png)

## 🚀 Features

- **✨ High-Quality Neural Style Transfer**: Powered by advanced Convolutional Neural Networks (CNNs) to deliver professional-grade style transfer results
- **🎨 Dynamic Style Blending**: Control the intensity of the transferred style with a responsive slider to achieve the perfect balance
- **⚡ Real-Time Performance**: Optimized for speed using PyTorch on GPU (when available), providing quick style transfer results
- **📱 Responsive Web Interface**: Beautiful, modern, and mobile-friendly design with smooth animations and gradients
- **🌐 AI-Powered Enhancement**: Includes experimental features for AI-driven image enhancement and detail preservation
- **💾 Easy Downloads**: Download your masterpieces in high resolution with a single click
- **📚 Educational Insights**: Built-in guide explaining the technology behind neural style transfer

## 🔧 Tech Stack

- **Framework**: Flask
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: Bootstrap 5, Custom Animations, Google Fonts
- **Machine Learning**: PyTorch, Torchvision
- **Models**: VGG16 (pretrained)

## 📂 Project Structure

```
styleForge/
├── Static/                     # Static frontend assets
│   ├── css/
│   ├── images/
│   └── js/
├── app.py                      # Main Flask application
├── content_data/             # Content images for training/demo
├── style_data/               # Style images for training/demo
├── examples/                 # Generated examples and model outputs
├── vgg_normalised.pth          # Pretrained VGG16 model weights
└── utils/                      # Utility scripts and model definitions
```

## 🏁 Getting Started

### Prerequisites

- Python 3.7+
- pip package installer
- PyTorch (GPU version recommended for faster performance)

### Installation

1. **Clone the repository** (if not already done)

   ```bash
   git clone <repository-url>
   cd styleForge
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   python app.py
   ```

4. **Open in Browser**

   Open your web browser and navigate to: `http://[IP_ADDRESS]`

## 🎨 How It Works

### 1. Content Image
Select an image that contains the main subject or composition you want to keep.

### 2. Style Image
Choose an image whose artistic style (colors, brushstrokes, textures) you want to apply.

### 3. Adjust Style Intensity
Use the slider to control how much of the style is applied:
- **Low (0-30%)**: Subtle style with most of the original content remaining
- **Medium (30-70%)**: Balanced fusion of content and style
- **High (70-100%)**: Dominant style with significant transformation

### 4. Generate
Click **Generate** to start the neural style transfer process. The AI will analyze both images and create a new masterpiece.

### 5. Download
Once the process is complete, download your unique creation in high quality.

## 🎛 Controls Reference

| Control | Description |
|---------|-------------|
| **Content Image** | Upload your base image |
| **Style Image** | Upload the style image you want to apply |
| **Style Intensity Slider** | Adjust style strength (0-100%) |
| **Generate Button** | Start style transfer process |
| **Download Button** | Save the generated image |

## 💻 Developer Guide

### Project Structure Explained

- **app.py**: Flask application entry point, handles routing, uploads, and processing
- **utils/models.py**: PyTorch model definitions (VGG Encoder/Decoder)
- **utils/stylize.py**: Core style transfer logic, loss functions, and optimization
- **static/**: Frontend assets, including custom CSS for the "Cyberpunk/Futuristic" theme
- **examples/**: Generated images from training runs and demos

### Adding New Styles

To add new styles to the application:
1. Place your style image in the `style_data/` directory
2. The style will automatically appear in the dropdown menu
3. Restart the Flask server

### Model Customization

For advanced users, you can customize the neural style transfer process:
- Adjust loss weights in `utils/stylize.py`
- Modify training parameters in `tran.py`
- Use different pretrained models by updating `utils/models.py`

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Submit a Pull Request
- Report bugs or suggest features
- Help improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Support

For issues, questions, or feature requests, please:
1. Open an issue on the GitHub repository
2. Contact the development team at [your-email@example.com]

## 🙏 Acknowledgments

- Thanks to [Gatys et al.](https://arxiv.org/abs/1508.06576) for the original Neural Style Transfer paper
- PyTorch and Torchvision community for the powerful deep learning tools
- Bootstrap and CSS community for the beautiful design components

---
**Built with ❤️ for AI Art Enthusiasts**
    
