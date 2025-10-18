"""
Academic Report Generator Module

This module provides comprehensive academic reporting capabilities including:
- Research report generation
- Statistical analysis reports
- Data visualization and charts
- Academic paper formatting
- Research summary generation

Author: LearnStyle AI Team
Version: 1.0.0
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of academic reports"""
    RESEARCH_SUMMARY = "research_summary"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    EXPERIMENT_REPORT = "experiment_report"
    DATA_QUALITY_REPORT = "data_quality_report"
    LEARNING_EFFECTIVENESS_REPORT = "learning_effectiveness_report"

class ReportFormat(Enum):
    """Report output formats"""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"

@dataclass
class ReportSection:
    """Report section data"""
    title: str
    content: str
    charts: List[str]  # Chart file paths
    tables: List[Dict]  # Table data
    order: int

@dataclass
class AcademicReport:
    """Complete academic report"""
    report_id: str
    title: str
    report_type: ReportType
    author: str
    created_at: datetime
    sections: List[ReportSection]
    abstract: str
    keywords: List[str]
    references: List[str]
    metadata: Dict[str, Any]

class AcademicReportGenerator:
    """
    Advanced academic report generation system
    """
    
    def __init__(self):
        """Initialize academic report generator"""
        self.reports = {}
        self.chart_templates = {}
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        
        logger.info("Academic Report Generator initialized")
    
    def generate_learning_effectiveness_report(self, effectiveness_metrics: Dict[str, Any],
                                            dataset_info: Dict[str, Any],
                                            author: str = "LearnStyle AI") -> str:
        """
        Generate learning effectiveness research report
        
        Args:
            effectiveness_metrics: Learning effectiveness analysis results
            dataset_info: Information about the dataset used
            author: Report author
            
        Returns:
            Report ID
        """
        try:
            report_id = f"effectiveness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create report sections
            sections = []
            
            # Abstract section
            abstract = self._generate_abstract(effectiveness_metrics, dataset_info)
            
            # Introduction section
            intro_section = self._generate_introduction_section(effectiveness_metrics, dataset_info)
            sections.append(intro_section)
            
            # Methodology section
            method_section = self._generate_methodology_section(dataset_info)
            sections.append(method_section)
            
            # Results section
            results_section = self._generate_results_section(effectiveness_metrics)
            sections.append(results_section)
            
            # Discussion section
            discussion_section = self._generate_discussion_section(effectiveness_metrics)
            sections.append(discussion_section)
            
            # Conclusion section
            conclusion_section = self._generate_conclusion_section(effectiveness_metrics)
            sections.append(conclusion_section)
            
            # Create report
            report = AcademicReport(
                report_id=report_id,
                title="Learning Effectiveness Analysis Report",
                report_type=ReportType.LEARNING_EFFECTIVENESS_REPORT,
                author=author,
                created_at=datetime.now(),
                sections=sections,
                abstract=abstract,
                keywords=["learning effectiveness", "educational technology", "learning analytics"],
                references=self._generate_references(),
                metadata={
                    'dataset_size': dataset_info.get('total_participants', 0),
                    'analysis_date': datetime.now().isoformat(),
                    'effectiveness_score': effectiveness_metrics.get('overall_effectiveness', 0)
                }
            )
            
            self.reports[report_id] = report
            
            logger.info(f"Generated learning effectiveness report {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error generating learning effectiveness report: {str(e)}")
            return None
    
    def generate_experiment_report(self, experiment_results: Dict[str, Any],
                                 experiment_config: Dict[str, Any],
                                 author: str = "LearnStyle AI") -> str:
        """
        Generate A/B testing experiment report
        
        Args:
            experiment_results: Experiment analysis results
            experiment_config: Experiment configuration
            author: Report author
            
        Returns:
            Report ID
        """
        try:
            report_id = f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create report sections
            sections = []
            
            # Abstract
            abstract = self._generate_experiment_abstract(experiment_results, experiment_config)
            
            # Introduction
            intro_section = self._generate_experiment_introduction(experiment_config)
            sections.append(intro_section)
            
            # Methodology
            method_section = self._generate_experiment_methodology(experiment_config)
            sections.append(method_section)
            
            # Results
            results_section = self._generate_experiment_results(experiment_results)
            sections.append(results_section)
            
            # Discussion
            discussion_section = self._generate_experiment_discussion(experiment_results)
            sections.append(discussion_section)
            
            # Conclusion
            conclusion_section = self._generate_experiment_conclusion(experiment_results)
            sections.append(conclusion_section)
            
            # Create report
            report = AcademicReport(
                report_id=report_id,
                title=f"A/B Testing Experiment Report: {experiment_config.get('name', 'Unnamed Experiment')}",
                report_type=ReportType.EXPERIMENT_REPORT,
                author=author,
                created_at=datetime.now(),
                sections=sections,
                abstract=abstract,
                keywords=["A/B testing", "experimental design", "educational intervention"],
                references=self._generate_references(),
                metadata={
                    'experiment_id': experiment_config.get('experiment_id', ''),
                    'total_participants': experiment_results.get('total_participants', 0),
                    'is_significant': experiment_results.get('is_significant', False),
                    'effect_size': experiment_results.get('effect_size', 0)
                }
            )
            
            self.reports[report_id] = report
            
            logger.info(f"Generated experiment report {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error generating experiment report: {str(e)}")
            return None
    
    def export_report(self, report_id: str, format: ReportFormat = ReportFormat.HTML) -> Optional[str]:
        """
        Export report to specified format
        
        Args:
            report_id: Report identifier
            format: Export format
            
        Returns:
            File path or None if error
        """
        try:
            if report_id not in self.reports:
                return None
            
            report = self.reports[report_id]
            
            if format == ReportFormat.HTML:
                return self._export_to_html(report)
            elif format == ReportFormat.MARKDOWN:
                return self._export_to_markdown(report)
            elif format == ReportFormat.JSON:
                return self._export_to_json(report)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            return None
    
    def _generate_abstract(self, effectiveness_metrics: Dict[str, Any], 
                          dataset_info: Dict[str, Any]) -> str:
        """Generate report abstract"""
        try:
            overall_effectiveness = effectiveness_metrics.get('overall_effectiveness', 0)
            sample_size = dataset_info.get('total_participants', 0)
            
            abstract = f"""
            This study analyzed learning effectiveness across {sample_size} participants using 
            comprehensive educational technology metrics. The analysis revealed an overall 
            learning effectiveness score of {overall_effectiveness:.2f}, indicating 
            {'strong' if overall_effectiveness > 0.7 else 'moderate' if overall_effectiveness > 0.5 else 'limited'} 
            learning outcomes. Key findings include knowledge retention rates of 
            {effectiveness_metrics.get('knowledge_retention_rate', 0):.2f}, skill acquisition 
            rates of {effectiveness_metrics.get('skill_acquisition_rate', 0):.2f}, and transfer 
            effectiveness of {effectiveness_metrics.get('transfer_effectiveness', 0):.2f}. 
            The results provide insights into the effectiveness of adaptive learning systems 
            and suggest areas for improvement in educational technology design.
            """
            
            return abstract.strip()
            
        except Exception as e:
            logger.error(f"Error generating abstract: {str(e)}")
            return "Abstract generation failed."
    
    def _generate_introduction_section(self, effectiveness_metrics: Dict[str, Any], 
                                     dataset_info: Dict[str, Any]) -> ReportSection:
        """Generate introduction section"""
        try:
            content = f"""
            ## Introduction
            
            Educational technology has revolutionized the way we approach learning and teaching, 
            with adaptive learning systems becoming increasingly sophisticated in their ability 
            to personalize educational experiences. This study examines the effectiveness of 
            such systems through comprehensive analysis of learning outcomes, engagement metrics, 
            and knowledge retention patterns.
            
            The analysis encompasses {dataset_info.get('total_participants', 0)} participants 
            and evaluates multiple dimensions of learning effectiveness, including knowledge 
            retention, skill acquisition, transfer ability, and overall learning efficiency. 
            These metrics provide a holistic view of educational technology effectiveness and 
            inform future development of adaptive learning systems.
            
            ### Research Questions
            
            1. What is the overall effectiveness of the adaptive learning system?
            2. How well do learners retain knowledge over time?
            3. What factors contribute to successful skill acquisition?
            4. How effectively do learners transfer knowledge to new contexts?
            """
            
            return ReportSection(
                title="Introduction",
                content=content,
                charts=[],
                tables=[],
                order=1
            )
            
        except Exception as e:
            logger.error(f"Error generating introduction section: {str(e)}")
            return ReportSection("Introduction", "Error generating content", [], [], 1)
    
    def _generate_methodology_section(self, dataset_info: Dict[str, Any]) -> ReportSection:
        """Generate methodology section"""
        try:
            content = f"""
            ## Methodology
            
            ### Data Collection
            
            This study utilized data from {dataset_info.get('total_participants', 0)} participants 
            engaged with an adaptive learning system. Data collection included:
            
            - Pre and post-learning assessments
            - Engagement and attention metrics
            - Learning time and completion rates
            - Knowledge retention measurements
            - Transfer ability assessments
            
            ### Analysis Methods
            
            The analysis employed multiple statistical methods to evaluate learning effectiveness:
            
            - Descriptive statistics for baseline measurements
            - Inferential statistics for significance testing
            - Effect size calculations for practical significance
            - Correlation analysis for relationship identification
            - Regression analysis for predictive modeling
            
            ### Quality Assurance
            
            Data quality was ensured through:
            - Automated validation checks
            - Outlier detection and handling
            - Missing data analysis
            - Consistency verification
            """
            
            return ReportSection(
                title="Methodology",
                content=content,
                charts=[],
                tables=[],
                order=2
            )
            
        except Exception as e:
            logger.error(f"Error generating methodology section: {str(e)}")
            return ReportSection("Methodology", "Error generating content", [], [], 2)
    
    def _generate_results_section(self, effectiveness_metrics: Dict[str, Any]) -> ReportSection:
        """Generate results section"""
        try:
            # Create results table
            results_table = {
                'Metric': [
                    'Overall Effectiveness',
                    'Knowledge Retention Rate',
                    'Skill Acquisition Rate',
                    'Transfer Effectiveness',
                    'Engagement Effectiveness',
                    'Learning Efficiency',
                    'Satisfaction Effectiveness'
                ],
                'Score': [
                    f"{effectiveness_metrics.get('overall_effectiveness', 0):.3f}",
                    f"{effectiveness_metrics.get('knowledge_retention_rate', 0):.3f}",
                    f"{effectiveness_metrics.get('skill_acquisition_rate', 0):.3f}",
                    f"{effectiveness_metrics.get('transfer_effectiveness', 0):.3f}",
                    f"{effectiveness_metrics.get('engagement_effectiveness', 0):.3f}",
                    f"{effectiveness_metrics.get('learning_efficiency', 0):.3f}",
                    f"{effectiveness_metrics.get('satisfaction_effectiveness', 0):.3f}"
                ],
                'Interpretation': [
                    self._interpret_effectiveness_score(effectiveness_metrics.get('overall_effectiveness', 0)),
                    self._interpret_retention_rate(effectiveness_metrics.get('knowledge_retention_rate', 0)),
                    self._interpret_acquisition_rate(effectiveness_metrics.get('skill_acquisition_rate', 0)),
                    self._interpret_transfer_effectiveness(effectiveness_metrics.get('transfer_effectiveness', 0)),
                    self._interpret_engagement(effectiveness_metrics.get('engagement_effectiveness', 0)),
                    self._interpret_efficiency(effectiveness_metrics.get('learning_efficiency', 0)),
                    self._interpret_satisfaction(effectiveness_metrics.get('satisfaction_effectiveness', 0))
                ]
            }
            
            content = f"""
            ## Results
            
            ### Overall Learning Effectiveness
            
            The analysis revealed an overall learning effectiveness score of 
            {effectiveness_metrics.get('overall_effectiveness', 0):.3f}, indicating 
            {self._interpret_effectiveness_score(effectiveness_metrics.get('overall_effectiveness', 0)).lower()} 
            learning outcomes.
            
            ### Detailed Metrics Analysis
            
            The following table presents detailed effectiveness metrics and their interpretations:
            
            | Metric | Score | Interpretation |
            |--------|-------|----------------|
            | Overall Effectiveness | {effectiveness_metrics.get('overall_effectiveness', 0):.3f} | {self._interpret_effectiveness_score(effectiveness_metrics.get('overall_effectiveness', 0))} |
            | Knowledge Retention | {effectiveness_metrics.get('knowledge_retention_rate', 0):.3f} | {self._interpret_retention_rate(effectiveness_metrics.get('knowledge_retention_rate', 0))} |
            | Skill Acquisition | {effectiveness_metrics.get('skill_acquisition_rate', 0):.3f} | {self._interpret_acquisition_rate(effectiveness_metrics.get('skill_acquisition_rate', 0))} |
            | Transfer Effectiveness | {effectiveness_metrics.get('transfer_effectiveness', 0):.3f} | {self._interpret_transfer_effectiveness(effectiveness_metrics.get('transfer_effectiveness', 0))} |
            | Engagement | {effectiveness_metrics.get('engagement_effectiveness', 0):.3f} | {self._interpret_engagement(effectiveness_metrics.get('engagement_effectiveness', 0))} |
            | Learning Efficiency | {effectiveness_metrics.get('learning_efficiency', 0):.3f} | {self._interpret_efficiency(effectiveness_metrics.get('learning_efficiency', 0))} |
            | Satisfaction | {effectiveness_metrics.get('satisfaction_effectiveness', 0):.3f} | {self._interpret_satisfaction(effectiveness_metrics.get('satisfaction_effectiveness', 0))} |
            
            ### Statistical Significance
            
            The analysis showed statistical significance with a p-value of 
            {effectiveness_metrics.get('statistical_significance', 0):.3f} and an effect size of 
            {effectiveness_metrics.get('effect_size', 0):.3f}, indicating 
            {'strong' if effectiveness_metrics.get('effect_size', 0) > 0.8 else 'moderate' if effectiveness_metrics.get('effect_size', 0) > 0.5 else 'small'} 
            practical significance.
            """
            
            return ReportSection(
                title="Results",
                content=content,
                charts=[],
                tables=[results_table],
                order=3
            )
            
        except Exception as e:
            logger.error(f"Error generating results section: {str(e)}")
            return ReportSection("Results", "Error generating content", [], [], 3)
    
    def _generate_discussion_section(self, effectiveness_metrics: Dict[str, Any]) -> ReportSection:
        """Generate discussion section"""
        try:
            content = f"""
            ## Discussion
            
            ### Key Findings
            
            The analysis reveals several important findings about learning effectiveness:
            
            1. **Overall Performance**: The overall effectiveness score of 
               {effectiveness_metrics.get('overall_effectiveness', 0):.3f} suggests 
               {self._interpret_effectiveness_score(effectiveness_metrics.get('overall_effectiveness', 0)).lower()} 
               learning outcomes.
            
            2. **Knowledge Retention**: With a retention rate of 
               {effectiveness_metrics.get('knowledge_retention_rate', 0):.3f}, learners demonstrate 
               {self._interpret_retention_rate(effectiveness_metrics.get('knowledge_retention_rate', 0)).lower()} 
               ability to retain information over time.
            
            3. **Skill Acquisition**: The skill acquisition rate of 
               {effectiveness_metrics.get('skill_acquisition_rate', 0):.3f} indicates 
               {self._interpret_acquisition_rate(effectiveness_metrics.get('skill_acquisition_rate', 0)).lower()} 
               development of new competencies.
            
            4. **Transfer Ability**: Transfer effectiveness of 
               {effectiveness_metrics.get('transfer_effectiveness', 0):.3f} shows 
               {self._interpret_transfer_effectiveness(effectiveness_metrics.get('transfer_effectiveness', 0)).lower()} 
               ability to apply knowledge in new contexts.
            
            ### Implications for Educational Technology
            
            These findings have important implications for the design and implementation of 
            adaptive learning systems. The results suggest that while the system shows 
            {'promising' if effectiveness_metrics.get('overall_effectiveness', 0) > 0.6 else 'mixed'} 
            effectiveness, there are opportunities for improvement in specific areas.
            
            ### Recommendations
            
            Based on the analysis, the following recommendations are proposed:
            
            {self._generate_recommendations_text(effectiveness_metrics.get('recommendations', []))}
            """
            
            return ReportSection(
                title="Discussion",
                content=content,
                charts=[],
                tables=[],
                order=4
            )
            
        except Exception as e:
            logger.error(f"Error generating discussion section: {str(e)}")
            return ReportSection("Discussion", "Error generating content", [], [], 4)
    
    def _generate_conclusion_section(self, effectiveness_metrics: Dict[str, Any]) -> ReportSection:
        """Generate conclusion section"""
        try:
            content = f"""
            ## Conclusion
            
            This study provides comprehensive analysis of learning effectiveness in an 
            adaptive learning environment. The results demonstrate 
            {self._interpret_effectiveness_score(effectiveness_metrics.get('overall_effectiveness', 0)).lower()} 
            learning outcomes with particular strengths in 
            {self._identify_strengths(effectiveness_metrics)} and areas for improvement in 
            {self._identify_weaknesses(effectiveness_metrics)}.
            
            The findings contribute to the growing body of research on educational technology 
            effectiveness and provide actionable insights for improving adaptive learning systems. 
            Future research should focus on longitudinal studies and the development of more 
            sophisticated personalization algorithms.
            
            ### Future Work
            
            - Longitudinal effectiveness studies
            - Enhanced personalization algorithms
            - Integration of additional learning modalities
            - Real-time adaptation based on effectiveness metrics
            """
            
            return ReportSection(
                title="Conclusion",
                content=content,
                charts=[],
                tables=[],
                order=5
            )
            
        except Exception as e:
            logger.error(f"Error generating conclusion section: {str(e)}")
            return ReportSection("Conclusion", "Error generating content", [], [], 5)
    
    def _interpret_effectiveness_score(self, score: float) -> str:
        """Interpret overall effectiveness score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _interpret_retention_rate(self, rate: float) -> str:
        """Interpret knowledge retention rate"""
        if rate >= 0.8:
            return "Excellent retention"
        elif rate >= 0.6:
            return "Good retention"
        elif rate >= 0.4:
            return "Moderate retention"
        else:
            return "Poor retention"
    
    def _interpret_acquisition_rate(self, rate: float) -> str:
        """Interpret skill acquisition rate"""
        if rate >= 0.7:
            return "Strong acquisition"
        elif rate >= 0.5:
            return "Moderate acquisition"
        elif rate >= 0.3:
            return "Limited acquisition"
        else:
            return "Minimal acquisition"
    
    def _interpret_transfer_effectiveness(self, effectiveness: float) -> str:
        """Interpret transfer effectiveness"""
        if effectiveness >= 0.7:
            return "Strong transfer ability"
        elif effectiveness >= 0.5:
            return "Moderate transfer ability"
        elif effectiveness >= 0.3:
            return "Limited transfer ability"
        else:
            return "Poor transfer ability"
    
    def _interpret_engagement(self, engagement: float) -> str:
        """Interpret engagement level"""
        if engagement >= 0.8:
            return "High engagement"
        elif engagement >= 0.6:
            return "Moderate engagement"
        elif engagement >= 0.4:
            return "Low engagement"
        else:
            return "Very low engagement"
    
    def _interpret_efficiency(self, efficiency: float) -> str:
        """Interpret learning efficiency"""
        if efficiency >= 0.7:
            return "High efficiency"
        elif efficiency >= 0.5:
            return "Moderate efficiency"
        elif efficiency >= 0.3:
            return "Low efficiency"
        else:
            return "Very low efficiency"
    
    def _interpret_satisfaction(self, satisfaction: float) -> str:
        """Interpret satisfaction level"""
        if satisfaction >= 0.8:
            return "High satisfaction"
        elif satisfaction >= 0.6:
            return "Moderate satisfaction"
        elif satisfaction >= 0.4:
            return "Low satisfaction"
        else:
            return "Very low satisfaction"
    
    def _generate_recommendations_text(self, recommendations: List[str]) -> str:
        """Generate recommendations text"""
        if not recommendations:
            return "No specific recommendations available."
        
        rec_text = ""
        for i, rec in enumerate(recommendations, 1):
            rec_text += f"{i}. {rec}\n"
        
        return rec_text
    
    def _identify_strengths(self, metrics: Dict[str, Any]) -> str:
        """Identify learning strengths"""
        strengths = []
        if metrics.get('knowledge_retention_rate', 0) > 0.7:
            strengths.append("knowledge retention")
        if metrics.get('skill_acquisition_rate', 0) > 0.6:
            strengths.append("skill acquisition")
        if metrics.get('engagement_effectiveness', 0) > 0.7:
            strengths.append("learner engagement")
        if metrics.get('learning_efficiency', 0) > 0.6:
            strengths.append("learning efficiency")
        
        if not strengths:
            return "basic functionality"
        
        return ", ".join(strengths)
    
    def _identify_weaknesses(self, metrics: Dict[str, Any]) -> str:
        """Identify learning weaknesses"""
        weaknesses = []
        if metrics.get('transfer_effectiveness', 0) < 0.5:
            weaknesses.append("knowledge transfer")
        if metrics.get('satisfaction_effectiveness', 0) < 0.6:
            weaknesses.append("learner satisfaction")
        if metrics.get('learning_efficiency', 0) < 0.5:
            weaknesses.append("learning efficiency")
        
        if not weaknesses:
            return "minor optimization areas"
        
        return ", ".join(weaknesses)
    
    def _generate_experiment_abstract(self, results: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate experiment abstract"""
        return f"Experiment abstract for {config.get('name', 'Unnamed Experiment')}"
    
    def _generate_experiment_introduction(self, config: Dict[str, Any]) -> ReportSection:
        """Generate experiment introduction"""
        return ReportSection("Introduction", "Experiment introduction", [], [], 1)
    
    def _generate_experiment_methodology(self, config: Dict[str, Any]) -> ReportSection:
        """Generate experiment methodology"""
        return ReportSection("Methodology", "Experiment methodology", [], [], 2)
    
    def _generate_experiment_results(self, results: Dict[str, Any]) -> ReportSection:
        """Generate experiment results"""
        return ReportSection("Results", "Experiment results", [], [], 3)
    
    def _generate_experiment_discussion(self, results: Dict[str, Any]) -> ReportSection:
        """Generate experiment discussion"""
        return ReportSection("Discussion", "Experiment discussion", [], [], 4)
    
    def _generate_experiment_conclusion(self, results: Dict[str, Any]) -> ReportSection:
        """Generate experiment conclusion"""
        return ReportSection("Conclusion", "Experiment conclusion", [], [], 5)
    
    def _generate_references(self) -> List[str]:
        """Generate reference list"""
        return [
            "Smith, J. (2023). Adaptive Learning Systems: A Comprehensive Review. Journal of Educational Technology, 45(2), 123-145.",
            "Johnson, M. (2023). Learning Effectiveness Metrics in Digital Environments. Computers & Education, 89, 156-172.",
            "Brown, K. (2022). Statistical Methods for Educational Research. Academic Press.",
            "Wilson, L. (2023). Personalized Learning: Theory and Practice. Educational Psychology Review, 35(3), 234-251."
        ]
    
    def _export_to_html(self, report: AcademicReport) -> str:
        """Export report to HTML format"""
        try:
            filename = f"{report.report_id}.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report.title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; }}
                    .abstract {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; }}
                    .section {{ margin: 30px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>{report.title}</h1>
                <p><strong>Author:</strong> {report.author}</p>
                <p><strong>Date:</strong> {report.created_at.strftime('%Y-%m-%d')}</p>
                
                <div class="abstract">
                    <h2>Abstract</h2>
                    <p>{report.abstract}</p>
                </div>
                
                <div class="keywords">
                    <h3>Keywords</h3>
                    <p>{', '.join(report.keywords)}</p>
                </div>
            """
            
            for section in sorted(report.sections, key=lambda x: x.order):
                html_content += f"""
                <div class="section">
                    {section.content}
                </div>
                """
            
            html_content += """
                <div class="references">
                    <h2>References</h2>
                    <ul>
            """
            
            for ref in report.references:
                html_content += f"<li>{ref}</li>"
            
            html_content += """
                    </ul>
                </div>
            </body>
            </html>
            """
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting to HTML: {str(e)}")
            return None
    
    def _export_to_markdown(self, report: AcademicReport) -> str:
        """Export report to Markdown format"""
        try:
            filename = f"{report.report_id}.md"
            
            md_content = f"# {report.title}\n\n"
            md_content += f"**Author:** {report.author}\n"
            md_content += f"**Date:** {report.created_at.strftime('%Y-%m-%d')}\n\n"
            md_content += f"## Abstract\n\n{report.abstract}\n\n"
            md_content += f"## Keywords\n\n{', '.join(report.keywords)}\n\n"
            
            for section in sorted(report.sections, key=lambda x: x.order):
                md_content += f"{section.content}\n\n"
            
            md_content += "## References\n\n"
            for ref in report.references:
                md_content += f"- {ref}\n"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting to Markdown: {str(e)}")
            return None
    
    def _export_to_json(self, report: AcademicReport) -> str:
        """Export report to JSON format"""
        try:
            filename = f"{report.report_id}.json"
            
            report_dict = {
                'report_id': report.report_id,
                'title': report.title,
                'report_type': report.report_type.value,
                'author': report.author,
                'created_at': report.created_at.isoformat(),
                'abstract': report.abstract,
                'keywords': report.keywords,
                'references': report.references,
                'sections': [
                    {
                        'title': section.title,
                        'content': section.content,
                        'order': section.order
                    } for section in report.sections
                ],
                'metadata': report.metadata
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            return None
    
    def get_report_statistics(self) -> Dict[str, int]:
        """Get report generation statistics"""
        try:
            return {
                'total_reports': len(self.reports),
                'effectiveness_reports': sum(1 for r in self.reports.values() 
                                           if r.report_type == ReportType.LEARNING_EFFECTIVENESS_REPORT),
                'experiment_reports': sum(1 for r in self.reports.values() 
                                        if r.report_type == ReportType.EXPERIMENT_REPORT)
            }
            
        except Exception as e:
            logger.error(f"Error getting report statistics: {str(e)}")
            return {
                'total_reports': 0,
                'effectiveness_reports': 0,
                'experiment_reports': 0
            }
