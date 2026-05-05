# Contributing Guidelines

Thank you for your interest in improving the Garbage Classifier project! This document provides guidelines for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and improve

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/garbage_classifier.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your branch: `git push origin feature/your-feature-name`
7. Open a Pull Request

## Development Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU (4GB+ VRAM for training)
- Windows 10/11
- Git

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For inference only
pip install -r requirements_inference_only.txt
```

## Areas for Contribution

### Model Improvements
- [ ] Improve accuracy on specific garbage types
- [ ] Reduce model size for faster inference
- [ ] Add new garbage categories
- [ ] Optimize for different hardware

### Code Quality
- [ ] Add unit tests
- [ ] Improve documentation
- [ ] Add type hints
- [ ] Optimize performance
- [ ] Fix bugs

### Deployment
- [ ] Add support for other OpenMV boards
- [ ] Improve OpenMV integration
- [ ] Add web dashboard
- [ ] Create mobile app integration

### Documentation
- [ ] Expand setup guides
- [ ] Add more examples
- [ ] Create video tutorials
- [ ] Document API usage

### Testing
- [ ] Develop test suite
- [ ] Add CI/CD pipeline
- [ ] Create benchmark tests
- [ ] Test on real hardware

## Pull Request Process

1. **Update documentation** - Ensure README and docs are updated
2. **Test thoroughly** - Run existing tests and add new ones
3. **Follow style guide** - Use consistent code formatting
4. **Keep commits clean** - Squash related commits
5. **Write clear messages** - Describe what and why

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe how you tested these changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Updated documentation
- [ ] Added/updated tests
- [ ] Tested on real hardware
```

## Coding Standards

### Python Style
- Follow PEP 8
- Use 4 spaces for indentation
- Add docstrings to functions
- Use type hints where possible

Example:
```python
def classify_garbage(image: np.ndarray, threshold: float = 0.5) -> Tuple[str, float]:
    """
    Classify garbage in an image.
    
    Args:
        image: Input image as numpy array
        threshold: Confidence threshold (0-1)
    
    Returns:
        Tuple of (class_name, confidence_score)
    """
    # Implementation
    pass
```

### MicroPython Style (OpenMV Scripts)
- Use clear variable names
- Keep functions small and focused
- Add comments for complex logic
- Optimize for memory constraints

## Testing

### Running Tests
```bash
python -m pytest tests/ -v
```

### Writing Tests
```python
import unittest
from src.classifier import Classifier

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier()
    
    def test_load_model(self):
        self.assertIsNotNone(self.classifier.model)
    
    def test_classification(self):
        # Add test logic
        pass
```

## Documentation

### Guidelines
- Use clear, simple language
- Include examples
- Add code snippets
- Update table of contents
- Include diagrams where helpful

### Markdown Formatting
```markdown
# Main Heading
## Subheading
### Sub-subheading

**Bold text** for emphasis
`code` for inline code
```python
# code blocks
```

## Commit Messages

Format: `type(scope): subject`

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation
- **style**: Formatting
- **refactor**: Code restructuring
- **perf**: Performance improvement
- **test**: Adding tests

Examples:
```
feat(classifier): add confidence threshold adjusting
fix(detector): resolve memory leak in inference loop
docs(deployment): update RT1062 setup guide
perf(model): optimize inference speed by 15%
```

## Performance Guidelines

### Model Training
- Document training parameters
- Record accuracy metrics
- Report inference time
- Include hardware specs

### Memory Usage
- Profile code for memory leaks
- Optimize for RT1062 (1MB RAM)
- Use efficient data structures
- Minimize model size

### Speed Optimization
- Target >2 FPS on RT1062
- Profile with `timeit`
- Avoid unnecessary file I/O
- Use hardware acceleration

## Release Process

1. Update version number
2. Update CHANGELOG
3. Create release tag
4. Write release notes
5. Deploy to PyPI (if applicable)

## Getting Help

- **Questions?** Open an issue with label `question`
- **Bug report?** Include:
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - Hardware/software specs
  - Error messages/logs
- **Feature request?** Describe:
  - What you need
  - Why you need it
  - Suggested implementation

## Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- GitHub Contributors page

## Legal

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).

---

**Thank you for contributing! Your efforts help make garbage classification better for everyone! 🎉**
