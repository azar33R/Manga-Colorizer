# Manga-Colorizer (Revamp) 🎨

Bring your black-and-white manga to life with state-of-the-art AI. This revamped version focus on speed, quality, and text preservation.

---

## 🚀 Key Improvements in this Version

- **NAFNet-S Denoiser**: Replaced legacy FFDNet for 3x faster denoising and superior image clarity.
- **CRAFT Text Preservation**: Automatically detects and protects manga text, ensuring dialogue remains crisp and 100% readable.
- **Smart Pipeline**: Automatically skips unnecessary processing (denoising clean images or upscaling high-res ones) to save time.
- **Post-Processing**: Integrated sharpening and adaptive saturation for vibrant, print-quality results.
- **High Performance**: Optimized for T4 GPUs (Colab/Kaggle), providing inference in ~130-190ms per image.

---

## ⚡ Quick Start (Server)

### Option A: Google Colab (Easiest)
1. Open the [Colab Script](Colab.txt).
2. Copy the code into a new [Google Colab](https://colab.research.google.com/) notebook.
3. Run all cells. It will provide a URL to paste into your extension.

### Option B: Local Hosting
1. **Clone & Install**:
   ```bash
   git clone -b revamp https://github.com/azar33R/Manga-Colorizer.git
   cd Manga-Colorizer/Backend
   pip install -r requirements.txt
   ```
2. **Download Models**: Place weights in `Backend/networks/`.
3. **Run**:
   ```bash
   python app-stream.py --preserve-text --post-process
   ```

---

## 🛠️ Client Setup (Extension)

1. **Install**:
   - **Chrome**: Load `Frontend-Chrome` folder as an "Unpacked Extension" in `chrome://extensions/`.
   - **Firefox**: Load `manifest.json` from `Frontend-Firefox` in `about:debugging`.
2. **Configure**:
   - Open the extension settings.
   - Paste your **Server URL** (from Colab or Localhost).
   - Click **Test** to verify connection.
3. **Enjoy**: Open any manga site, click **Add to Sites**, and hit **Colorize!**.

---

## 📦 Pro Version (No Setup)
For a seamless, fully managed experience with zero setup, check out **MangaColorizerPro**:
- [Chrome Web Store](https://chromewebstore.google.com/detail/mangacolorizerpro/ofeggeimdlfipekkabopihemnefgkapk)
- [Firefox Add-ons](https://addons.mozilla.org/en-US/firefox/addon/mangacolorizerpro/)

---

## 📜 Credits
- **AI Models**: AlacGAN, CycleGAN, NAFNet, RealESRGAN.
- **Contributors**: Based on work by qweasdd, xiaogdgenuine, xinntao, and vatavian.
