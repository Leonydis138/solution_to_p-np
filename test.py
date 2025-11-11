#!/bin/bash
# Complete automated setup for Circuit Lower Bounds Research Platform
# Run this script to set up everything automatically

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Circuit Lower Bounds Research Platform - Full Setup     ║"
echo "║                   Automated Installation                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Check Python
echo -e "${YELLOW}[1/8] Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓ Python $python_version found${NC}"
    
    # Check version is 3.9+
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
        echo -e "${GREEN}✓ Python version OK (3.9+)${NC}"
    else
        echo -e "${RED}✗ Python 3.9+ required, found $python_version${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Step 2: Check/Install pip
echo -e "\n${YELLOW}[2/8] Checking pip...${NC}"
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓ pip found${NC}"
else
    echo -e "${YELLOW}Installing pip...${NC}"
    python3 -m ensurepip --upgrade
fi

# Step 3: Create virtual environment
echo -e "\n${YELLOW}[3/8] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Remove and recreate? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    else
        echo -e "${GREEN}✓ Using existing virtual environment${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Step 4: Activate virtual environment
echo -e "\n${YELLOW}[4/8] Activating virtual environment...${NC}"
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Step 5: Upgrade pip
echo -e "\n${YELLOW}[5/8] Upgrading pip, setuptools, wheel...${NC}"
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓ Core tools upgraded${NC}"

# Step 6: Install dependencies
echo -e "\n${YELLOW}[6/8] Installing dependencies...${NC}"
echo -e "${BLUE}This may take 3-5 minutes. Please wait...${NC}"

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}Creating requirements.txt...${NC}"
    cat > requirements.txt << 'EOF'
numpy==1.24.3
pandas==2.0.3
scipy==1.11.4
matplotlib==3.8.0
seaborn==0.13.0
streamlit==1.29.0
plotly==5.18.0
statsmodels==0.14.0
scikit-learn==1.3.2
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0
tqdm==4.66.1
joblib==1.3.2
python-dotenv==1.0.0
openpyxl==3.1.2
xlsxwriter==3.1.9
EOF
fi

# Install core packages first (most likely to have issues)
pip install numpy==1.24.3 --quiet && echo -e "${GREEN}  ✓ numpy${NC}"
pip install pandas==2.0.3 --quiet && echo -e "${GREEN}  ✓ pandas${NC}"
pip install scipy==1.11.4 --quiet && echo -e "${GREEN}  ✓ scipy${NC}"

# Install remaining packages
pip install -r requirements.txt --quiet && echo -e "${GREEN}  ✓ All dependencies${NC}"

echo -e "${GREEN}✓ All packages installed${NC}"

# Step 7: Verify installation
echo -e "\n${YELLOW}[7/8] Verifying installation...${NC}"

verification_failed=0

check_package() {
    if python -c "import $1" 2>/dev/null; then
        version=$(python -c "import $1; print($1.__version__)" 2>/dev/null || echo "OK")
        echo -e "${GREEN}  ✓ $1 ($version)${NC}"
        return 0
    else
        echo -e "${RED}  ✗ $1 - FAILED${NC}"
        return 1
    fi
}

check_package "numpy" || verification_failed=1
check_package "pandas" || verification_failed=1
check_package "scipy" || verification_failed=1
check_package "matplotlib" || verification_failed=1
check_package "streamlit" || verification_failed=1
check_package "plotly" || verification_failed=1
check_package "statsmodels" || verification_failed=1
check_package "sklearn" || verification_failed=1

if [ $verification_failed -eq 1 ]; then
    echo -e "\n${RED}✗ Some packages failed to install${NC}"
    echo -e "${YELLOW}Try running: pip install -r requirements.txt${NC}"
    exit 1
fi

# Step 8: Create project structure
echo -e "\n${YELLOW}[8/8] Creating project structure...${NC}"

mkdir -p data/raw data/processed data/examples
mkdir -p results
mkdir -p logs
mkdir -p docs/paper/figures
mkdir -p src/validation src/analysis src/utils
mkdir -p tests
mkdir -p .streamlit

echo -e "${GREEN}✓ Project directories created${NC}"

# Create .streamlit/config.toml
if [ ! -f ".streamlit/config.toml" ]; then
    cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
EOF
    echo -e "${GREEN}✓ Streamlit config created${NC}"
fi

# Create minimal test app if streamlit_app.py doesn't exist
if [ ! -f "streamlit_app.py" ]; then
    echo -e "${YELLOW}Creating minimal test app...${NC}"
    cat > streamlit_app.py << 'EOF'
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Circuit Lower Bounds", page_icon="🔬", layout="wide")

st.markdown("# 🔬 Circuit Lower Bounds Research Platform")
st.markdown("## Installation Successful! ✅")

st.success("""
All dependencies are installed correctly!

You can now:
1. Explore the demo below
2. Replace this file with the full application
3. Start running validations
""")

# Simple demo
st.markdown("---")
st.markdown("### Quick Demo")

n = st.slider("Variables per component (n)", 10, 100, 20)
t = st.slider("Number of components (t)", 2, 20, 5)

if st.button("Run Test"):
    data = np.random.randn(100)
    fig = go.Figure(data=[go.Histogram(x=data)])
    fig.update_layout(title="Test Visualization")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ Test successful with n={n}, t={t}")

st.markdown("---")
st.info("💡 Replace this file with the full streamlit_app.py to access all features")
EOF
    echo -e "${GREEN}✓ Test app created${NC}"
fi

# Final summary
echo -e "\n${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  🎉 Setup Complete! 🎉                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Installation Summary:${NC}"
echo -e "  ${GREEN}✓${NC} Python $(python --version 2>&1 | awk '{print $2}')"
echo -e "  ${GREEN}✓${NC} Virtual environment: venv/"
echo -e "  ${GREEN}✓${NC} All dependencies installed"
echo -e "  ${GREEN}✓${NC} Project structure created"
echo -e "  ${GREEN}✓${NC} Configuration files ready"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  ${BLUE}1.${NC} Activate the virtual environment:"
echo -e "     ${GREEN}source venv/bin/activate${NC}"
echo -e "\n  ${BLUE}2.${NC} Run the application:"
echo -e "     ${GREEN}streamlit run streamlit_app.py${NC}"
echo -e "\n  ${BLUE}3.${NC} Open your browser to:"
echo -e "     ${GREEN}http://localhost:8501${NC}"

echo -e "\n${YELLOW}Troubleshooting:${NC}"
echo -e "  ${BLUE}•${NC} If you see 'No module named X': ${GREEN}pip install X${NC}"
echo -e "  ${BLUE}•${NC} If port 8501 is in use: ${GREEN}streamlit run streamlit_app.py --server.port 8502${NC}"
echo -e "  ${BLUE}•${NC} For help: ${GREEN}streamlit run streamlit_app.py --help${NC}"

echo -e "\n${GREEN}Ready to validate circuit lower bounds! 🚀${NC}\n"

# Auto-run option
echo -e "${YELLOW}Start the application now? (y/n)${NC}"
read -r start_now

if [[ "$start_now" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "\n${GREEN}Starting Streamlit...${NC}"
    streamlit run streamlit_app.py
fi
