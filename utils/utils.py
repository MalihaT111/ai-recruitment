import re
import os
import scipy.stats as stats
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import pickle 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------
# NLTK setup
# -----------------------
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------
# Preprocessing
# -----------------------
def preprocess_text(text):
    """Lowercase, tokenize, remove stopwords, and lemmatize."""
    if pd.isna(text):
        return ""
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)


# -----------------------
# Domain Detection
# -----------------------
# -----------------------
# Refined Domain Detection
# -----------------------
# -----------------------
# Unified Domain Keywords (Aligned with DOMAIN_GROUPS)
# -----------------------
DOMAIN_KEYWORDS_REFERENCE = {
    "Tech & IT": [
        'programming', 'software engineering', 'software development', 'python', 'java', 'c++',
        'sql', 'api', 'database', 'web development', 'frontend', 'backend',
        'data science', 'machine learning', 'artificial intelligence', 'deep learning',
        'devops', 'docker', 'kubernetes', 'cloud computing', 'aws', 'azure', 'gcp',
        'network administration', 'cybersecurity', 'linux', 'git', 'automation', 'debugging'
    ],

    "Finance & Accounting": [
        'finance', 'accounting', 'bookkeeping', 'audit', 'taxation', 'financial reporting',
        'budget forecast', 'cash flow', 'balance sheet', 'ledger', 'valuation',
        'accounts payable', 'accounts receivable', 'profit loss', 'treasury',
        'banking operations', 'credit risk', 'capital markets', 'investment analysis',
        'financial modeling', 'cost analysis', 'invoice', 'reconciliation', 'economics'
    ],

    "Business & Sales": [
        'business development', 'sales strategy', 'b2b', 'b2c', 'client acquisition',
        'cold calling', 'negotiation', 'lead generation', 'pipeline management',
        'revenue growth', 'sales forecasting', 'crm', 'prospecting', 'partnerships',
        'account executive', 'quotas', 'upselling', 'cross-selling', 'closing deals'
    ],

    "Law & Advocacy": [
        'legal', 'attorney', 'lawyer', 'litigation', 'legal research', 'contract law',
        'corporate law', 'civil law', 'criminal law', 'intellectual property',
        'regulatory compliance', 'court', 'case management', 'legal drafting'
    ],

    "Healthcare": [
        'healthcare', 'medical', 'nurse', 'clinic', 'patient care', 'diagnosis',
        'treatment', 'therapy', 'hospital', 'physician', 'pharmacy', 'rehabilitation',
        'fitness', 'nutrition', 'public health', 'wellness', 'surgery', 'medication'
    ],

    "HR & Operations": [
        'human resources', 'recruitment', 'recruiter', 'talent acquisition',
        'employee engagement', 'onboarding', 'offboarding', 'payroll',
        'benefits administration', 'performance appraisal', 'training development',
        'conflict resolution', 'hr policies', 'labor relations', 'workforce planning',
        'compliance', 'organizational development', 'hr analytics', 'diversity inclusion',
        'operations management', 'administrative', 'vendor management', 'logistics',
        'inventory management', 'documentation'
    ],

    "Creative & Design": [
        'graphic design', 'illustrator', 'photoshop', 'branding', 'typography',
        'ux design', 'ui design', 'web design', 'layout', 'content creation',
        'copywriting', 'animation', 'video editing', 'photography', 'creative direction',
        'visual storytelling', 'marketing collateral', 'digital media', 'social media campaign'
    ],

    "Education": [
        'teacher', 'professor', 'lecturer', 'education', 'curriculum development',
        'classroom management', 'student engagement', 'lesson planning',
        'academic advisor', 'training instructor', 'tutoring', 'pedagogy', 'educational leadership'
    ],

    "Manufacturing & Construction": [
        'construction', 'engineering', 'project planning', 'blueprint', 'contractor',
        'civil engineering', 'structural', 'mechanical', 'electrical', 'site management',
        'quality assurance', 'manufacturing', 'production line', 'machinery', 'safety compliance'
    ],

    "Agriculture & Environment": [
        'agriculture', 'farming', 'crop', 'soil', 'sustainability', 'irrigation',
        'environmental management', 'ecology', 'forestry', 'conservation',
        'organic farming', 'livestock', 'rural development', 'water resources'
    ],

    "Hospitality & Food": [
        'chef', 'kitchen', 'menu planning', 'catering', 'hospitality', 'food safety',
        'culinary arts', 'restaurant operations', 'inventory control', 'sanitation',
        'banquet', 'hotel management', 'customer service', 'guest relations'
    ],

    "Other Services": [
        'customer support', 'bpo', 'client service', 'call center', 'aviation',
        'airline', 'flight attendant', 'pilot', 'maintenance', 'fashion', 'apparel',
        'retail', 'merchandising', 'event planning'
    ]
}


# Weighted keyword matching
def infer_keyword_domain(text, keywords_dict=DOMAIN_KEYWORDS_REFERENCE):
    """Return the most likely domain based on weighted keyword occurrences."""
    text = str(text).lower()
    scores = {}

    for domain, keywords in keywords_dict.items():
        score = 0
        for kw in keywords:
            if kw in text:
                # Give higher weight to multi-word, domain-specific terms
                weight = 2 if len(kw.split()) > 1 else 1
                score += weight
        scores[domain] = score

    return max(scores, key=scores.get) if max(scores.values()) > 0 else "other"

# DOMAIN_KEYWORDS_REFERENCE = {
#     'hr': [
#         'human resources', 'hr', 'recruitment', 'recruiting', 'hiring',
#         'payroll', 'benefits', 'employee relations', 'compensation',
#         'performance management', 'talent acquisition', 'training',
#         'onboarding', 'diversity', 'compliance', 'employee engagement',
#         'career development', 'hr policies', 'conflict resolution',
#         'organizational development', 'leadership', 'communication skills',
#         'workplace safety', 'microsoft office', 'workforce planning',
#         'hr analytics', 'labor law', 'employee retention'
#     ],
#     'finance': [
#         'finance', 'financial', 'accounting', 'budget', 'budgeting', 'audit', 'tax',
#         'bookkeeping', 'financial analysis', 'forecasting', 'financial modeling',
#         'cash flow', 'profit', 'loss', 'ledger', 'accounts payable',
#         'accounts receivable', 'payable', 'receivable', 'valuation',
#         'cost analysis', 'financial reporting', 'economics', 'treasury',
#         'capital markets', 'credit', 'debit', 'banking', 'investment',
#         'excel', 'power bi', 'data analysis'
#     ],
#     'it': [
#         'programming', 'software', 'development', 'software development',
#         'software engineer', 'python', 'java', 'sql', 'database', 'web development',
#         'network', 'system administration', 'cloud', 'aws', 'azure', 'gcp',
#         'devops', 'docker', 'kubernetes', 'linux', 'git', 'version control',
#         'testing', 'debugging', 'api', 'backend', 'frontend', 'node.js', 'react',
#         'data science', 'machine learning', 'tensorflow', 'automation',
#         'cybersecurity'
#     ],
#     'sales': [
#         'sales', 'business development', 'account management', 'revenue', 'crm',
#         'client', 'customer', 'lead generation', 'cold calling', 'prospecting',
#         'presentation', 'closing deals', 'negotiation', 'pipeline', 'quota',
#         'target', 'territory', 'upselling', 'cross-selling', 'b2b', 'b2c',
#         'account executive', 'retail', 'merchandising', 'promotion', 'marketing',
#         'sales strategy', 'partnerships', 'client relations', 'relationship management',
#         'sales operations', 'business partnerships'
#     ],
#     'administration': [
#         'administrative', 'secretary', 'assistant', 'coordination', 'office',
#         'organization', 'communication', 'customer service', 'documentation',
#         'inventory', 'scheduling', 'calendar management', 'data entry', 'filing',
#         'record keeping', 'reception', 'travel arrangements', 'correspondence',
#         'procurement', 'clerical', 'executive assistant', 'meeting planning',
#         'office management', 'support staff', 'event planning', 'vendor management',
#         'front desk', 'logistics', 'supplies', 'records management',
#         'budget tracking', 'document control', 'office coordination',
#         'front office', 'mail management'
#     ],
#     'research': [
#         'research', 'analyst', 'analysis', 'data analysis', 'methodology',
#         'report', 'evaluation', 'literature review', 'hypothesis', 'experiment',
#         'survey', 'study', 'quantitative', 'qualitative', 'statistics', 'modeling',
#         'scientific', 'investigation', 'findings', 'insight', 'insights',
#         'publication', 'predictive modeling', 'data visualization',
#         'policy analysis', 'impact assessment', 'data collection', 'r',
#         'spss', 'tableau', 'power bi'
#     ]
# }
DOMAIN_GROUPS = {
    "Tech & IT": ["INFORMATION-TECHNOLOGY", "ENGINEERING"],
    "Finance & Accounting": ["FINANCE", "ACCOUNTANT", "BANKING"],
    "Business & Sales": ["BUSINESS-DEVELOPMENT", "SALES", "CONSULTANT"],
    "Law & Advocacy": ["ADVOCATE"],
    "Healthcare": ["HEALTHCARE", "FITNESS"],
    "HR & Operations": ["HR", "PUBLIC-RELATIONS"],
    "Creative & Design": ["DESIGNER", "ARTS", "DIGITAL-MEDIA"],
    "Education": ["TEACHER"],
    "Manufacturing & Construction": ["CONSTRUCTION", "AUTOMOBILE"],
    "Agriculture & Environment": ["AGRICULTURE"],
    "Hospitality & Food": ["CHEF"],
    "Other Services": ["APPAREL", "BPO", "AVIATION"],
}
def map_domain(category):
    """Map detailed resume categories to broader domain clusters."""
    for domain, cats in DOMAIN_GROUPS.items():
        if category in cats:
            return domain
    return "Other"

def attach_domain_labels(df, category_col="Category"):
    """Attach broader domain labels to resumes."""
    df["DomainCluster"] = df[category_col].apply(map_domain)
    return df

# def detect_domain(text, domain_keywords=domain_keywords):
#     text_lower = text.lower()
#     scores = {domain: sum(kw in text_lower for kw in keywords)
#               for domain, keywords in domain_keywords.items()}
#     return max(scores, key=scores.get) if max(scores.values()) > 0 else "other"

# -----------------------
# Scoring parameters
# -----------------------
weights = {
    'skills': 0.35,
    'experience': 0.20,
    'education': 0.15,
    'semantic': 0.15,
    'domain': 0.15
}

education_levels = {'phd': 4, 'master': 3, 'bachelor': 2, 'associate': 1, 'diploma': 0.5}
skills = ['excel', 'word', 'powerpoint', 'sql', 'python', 'project management', 'data analysis', 'ms office', 'microsoft office']
experience_words = ['manager', 'director', 'senior', 'lead', 'specialist', 'analyst']





