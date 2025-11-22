# Documentation Index

Comprehensive documentation for the MIMIC-CXR Unsupervised Anomaly Detection preprocessing pipeline.

## Quick Navigation

### For Developers and Architects

**[ARCHITECTURE.md](ARCHITECTURE.md)** - Deep technical architecture documentation
- Base processor pattern and class hierarchy
- Module organization and dependencies
- Design decisions and rationale (full resolution preservation, NOT_DONE tokens, etc.)
- Performance characteristics and bottlenecks
- Mermaid diagrams for class hierarchy and data flow
- Extension points for adding new processors/features

**Best for**: Understanding the codebase, extending functionality, troubleshooting

### For Data Scientists and ML Engineers

**[DATA_SCHEMA.md](DATA_SCHEMA.md)** - Complete data schema reference
- Cohort CSV schema (all 28 columns with types, descriptions, examples)
- Structured features JSON schema with field explanations
- Text features tensor format
- Image tensor specifications
- Relationships between files and loading patterns
- Example records for each format

**Best for**: Loading and using preprocessed data, understanding feature formats

### For Users and Operators

**[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Configuration decision guide
- Decision trees for key settings (image resolution, Claude usage, encoding methods)
- Explanation of each config section (image, text, structured, processing)
- Tradeoffs tables (speed vs quality, cost vs completeness)
- Recommended presets (fast mode, quality mode, balanced mode, debug mode)
- Performance tuning guide
- Common scenarios and solutions

**Best for**: Configuring the pipeline, optimizing performance, cost management

## Documentation Statistics

| Document | Lines | Size | Code Examples | Diagrams |
|----------|-------|------|---------------|----------|
| ARCHITECTURE.md | 880 | 26 KB | 15+ | 4 Mermaid |
| DATA_SCHEMA.md | 907 | 26 KB | 20+ | 0 |
| CONFIGURATION_GUIDE.md | 1,107 | 26 KB | 25+ | 5 Mermaid |
| **Total** | **2,894** | **78 KB** | **60+** | **9** |

## Quick Start Guide

### 1. First-Time Users

Start here:
1. Read the main [README.md](../README.md) for quick start and basic usage
2. Follow [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) → "Recommended Presets" → "Balanced Mode"
3. Check [DATA_SCHEMA.md](DATA_SCHEMA.md) → "Example Records" to understand outputs

### 2. Developers

Start here:
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) → "Base Processor Pattern"
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) → "Module Organization"
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) → "Extension Points" for adding features

### 3. Troubleshooting

Common issues:
1. **Configuration errors**: [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) → "Configuration Validation"
2. **Performance issues**: [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) → "Performance Tuning"
3. **Data loading errors**: [DATA_SCHEMA.md](DATA_SCHEMA.md) → "File Relationships and Loading"
4. **Architecture questions**: [ARCHITECTURE.md](ARCHITECTURE.md) → "Design Decisions and Rationale"

## Key Topics by Document

### ARCHITECTURE.md

- **Base Processor Pattern**: Abstract base classes for all processors
- **Class Hierarchy**: Mermaid diagram showing inheritance structure
- **Data Flow**: End-to-end pipeline flow with diagrams
- **Design Rationale**: Why full resolution? Why NOT_DONE tokens?
- **Performance**: Bottlenecks and optimization strategies
- **Testing**: Unit test architecture (60+ tests)

### DATA_SCHEMA.md

- **Cohort CSV**: 28 columns with complete specifications
- **Image Tensors**: Shape, dtype, normalization, loading examples
- **Structured Features**: JSON schema, NOT_DONE token format, temporal features
- **Text Features**: Claude summaries, ClinicalBERT tokens, entity extraction
- **Loading Patterns**: PyTorch Dataset integration examples
- **Example Records**: Real-world sample data

### CONFIGURATION_GUIDE.md

- **Decision Trees**: Interactive flowcharts for configuration choices
- **Configuration Sections**: Detailed explanation of each YAML section
- **Presets**: Fast, Balanced, Quality, Debug, Inference modes
- **Tradeoffs**: Speed vs Quality, Cost vs Completeness tables
- **Performance Tuning**: GPU usage, parallelization, memory optimization
- **Common Scenarios**: 7 real-world configuration scenarios

## Additional Resources

### Internal Documentation

- [Main README](../README.md) - Quick start, installation, basic usage
- [Step 2 README](../step2_preprocessing/README.md) - Detailed Step 2 documentation
- [Testing README](../step2_preprocessing/tests/README.md) - Test suite overview
- [Notebooks README](../step2_preprocessing/notebooks/README.md) - Jupyter notebooks

### External References

**MIMIC Datasets**:
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) - Chest X-ray database
- [MIMIC-IV](https://physionet.org/content/mimiciv/) - Hospital EHR database
- [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) - Emergency department database

**Tools and Models**:
- [scispacy](https://allenai.github.io/scispacy/) - Medical NER
- [ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) - Clinical text embeddings
- [Claude](https://www.anthropic.com/claude) - LLM for summarization
- [LangChain](https://python.langchain.com/) - LLM orchestration framework

## Contributing to Documentation

When updating documentation, maintain:

1. **Consistency**: Use same terminology across all docs
2. **Examples**: Include code examples for all major concepts
3. **Cross-references**: Link related sections across documents
4. **Diagrams**: Use Mermaid for visual explanations
5. **Tables**: Use tables for comparing options/tradeoffs

### Documentation Standards

- **Code blocks**: Use syntax highlighting (```python, ```yaml, ```bash)
- **Headers**: Use ## for main sections, ### for subsections
- **Links**: Use relative links for internal docs, absolute for external
- **Examples**: Real examples from actual pipeline runs
- **Updates**: Keep version numbers and statistics current

## Version History

- **November 2025**: Initial comprehensive documentation release
  - ARCHITECTURE.md: 880 lines
  - DATA_SCHEMA.md: 907 lines
  - CONFIGURATION_GUIDE.md: 1,107 lines
  - Total: 2,894 lines of documentation

## Feedback

For documentation issues or suggestions:
1. Open an issue in the project repository
2. Tag with `documentation` label
3. Reference specific document and section

---

**Last Updated**: November 22, 2025
