# Circuit Lower Bounds Explorer

## Overview

An interactive web application for validating and visualizing circuit lower bounds in computational complexity theory. The platform focuses on composition theorems and adversarial restrictions, providing statistical validation of theoretical results through Monte Carlo simulations and hypothesis testing. Built with Streamlit for rapid prototyping and interactive exploration of circuit complexity bounds.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
**Technology**: Streamlit web framework
- **Rationale**: Enables rapid development of interactive data applications with minimal boilerplate
- **Trade-offs**: Simplified deployment and built-in state management, but limited customization compared to traditional web frameworks
- **Page Configuration**: Wide layout with custom page title and icon for professional presentation

### Visualization Layer
**Technologies**: Plotly (interactive charts), Matplotlib (static exports)
- **Plotly Graph Objects & Express**: Primary visualization for interactive exploration
  - Enables dynamic user interaction with complexity data
  - Export capabilities for research publication
- **Matplotlib**: Fallback for PDF report generation
  - Configured with 'Agg' backend for server-side rendering without display
  - Ensures compatibility in headless environments

### Statistical Validation Engine
**Core Libraries**: NumPy, SciPy
- **Monte Carlo Simulation**: Generates random test cases for circuit behavior validation
- **Hypothesis Testing**: Multiple statistical tests per configuration:
  - Cross-influence tests using exponential distributions and normal CDFs
  - Concentration tests using beta distributions
  - Product composition tests
  - KKL-type influence tests
  - Monotone circuit tests
- **Statistical Rigor**: 
  - Confidence levels (default 99%)
  - Bootstrap sampling (configurable, default 1000 samples)
  - P-value thresholds (0.01) for test validation
  - Adaptive trial counts (200-1000 range)

### Data Management
**Format**: JSON-based configuration and results storage
- **Configuration Management**: Dataclass-based validation configurations with type safety
- **Results Persistence**: Timestamped validation results with full metadata
- **File Organization**:
  - `figure_data.json`: Pre-computed visualization datasets
  - `validation_results.json`: Summary statistics
  - `supplementary.json`: Additional research materials
  - Dynamic results in `validation_results/` directory

### Report Generation
**Technology**: ReportLab PDF library
- **Purpose**: Export validation results and visualizations for academic publication
- **Components**:
  - Document templates with professional styling
  - Table generation for statistical results
  - Embedded figure support
  - Multi-page layout with proper spacing and breaks
- **Trade-off**: More complex than simple HTML export, but provides publication-ready PDF output

### Computational Model
**Domain**: Circuit Complexity Theory
- **Independent Circuits**: Simulations for circuits with independent components
- **Dependent Circuits**: Validation of circuits with cross-component dependencies
- **Parameters**:
  - `n`: Input size/circuit width
  - `t`: Circuit depth/time parameter
- **Validation Approach**: Statistical hypothesis testing on randomly generated circuit configurations to verify theoretical lower bounds

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework and UI components
- **numpy**: Numerical computing and array operations for simulation data
- **scipy.stats**: Statistical distributions and hypothesis testing functions
- **plotly**: Interactive visualization (graph_objects and express modules)
- **pandas**: Data manipulation and tabular data handling
- **reportlab**: PDF generation for academic reports
- **matplotlib**: Static plot generation and export

### Data Storage
- **File System**: Local JSON files for configuration and results
  - No external database currently used
  - Results stored in structured directory (`validation_results/`)
  
### Third-Party Services
- **Mathpix**: Font rendering for mathematical notation (referenced in HTML assets)
- **CDN**: External font delivery for consistent mathematical typography

### Development Artifacts
- **main.py**: Minimal entry point (likely for Replit environment detection)
- **attached_assets/**: Research paper artifacts and demonstration materials
  - LaTeX source references
  - HTML rendering examples
  - Pre-generated validation datasets