import re
import numpy as np
from collections import Counter

# ============================================
# STOPWORDS (Your custom list)
# ============================================
custom_stopwords = {
    "state","city","company","name","university","office","llc","cjsc",
    "armenia","skills","experience","team","work","information",
    "professional","current","high","including","responsible","managed",
    "maintained","developed","ensure","quality","customers","customer",
    "service","support","working","activities","projects.","motivated",
    "highly","qualified","looking","seeking","candidate","incumbent",
    "position","provide","well","business","management","project",
    "development","training","sales","financial","marketing","design",
    "product","system","process","staff","manager","new","education",
    "microsoft","client","clients","provided","services","program",
    "school","college","daily","monthly","time","within","systems",
    "various","account","accounts","accounting","reports","summary",
    "performance","maintain","created","knowledge","worked","general",
    "strong","public","projects","technology","employee","health",
    "food","operations","senior","experienced","applications",
    "international","supervision","person","part","based","also",
    "manage","related","direct","different","company's","company.",
    "manager.","team.","products","students","january","february",
    "march","april","may","june","july","august","september","october",
    "november","december","using","communication","assisted","years",
    "department","prepared","needs","excellent","cash","student",
    "member","multiple","sales,","2014","he/","candidates","role",
    "join","fulfill","successful","applications.","development.",
    "responsibility","perform","providing","armenia.","armenian",
    "yerevan", "assistant", "lead", "key", "2013", "programs", "assist",
    "design,", "products.", "take", "head", "involved", "center",
    "long-term","performed","issues","internal","ability","personal",
    "special","annual","cost","companys","ensuring","leadership",
    "community","human","employees","control","primary","2012","help",
    "managing","national"
}


# ============================================
# DOMAIN KEYWORDS (Your improved full set)
# ============================================
DOMAIN_KEYWORDS = {
    "Tech & IT": [
        "software engineer","software developer","backend","frontend",
        "full stack","api","devops","cloud","aws","azure","gcp","database",
        "sql","python","java","javascript","c++","cybersecurity","network",
        "system administration","data science","machine learning","ai",
        "artificial intelligence","automation","docker","kubernetes","linux",
        "debugging","testing"
    ],
    "Finance & Accounting": [
        "finance","financial analyst","accounting","audit","bookkeeping",
        "taxation","ledger","reconciliation","accounts payable","accounts receivable",
        "invoice","budgeting","forecasting","balance sheet","profit and loss",
        "cost analysis","valuation","treasury","capital markets","investment analysis",
        "economics"
    ],
    "Business & Sales": [
        "business development","sales","b2b","b2c","client acquisition","crm",
        "pipeline management","lead generation","negotiation","quota","revenue growth",
        "upselling","cross selling","account executive","partnerships","closing deals"
    ],
    "Law & Advocacy": [
        "legal","attorney","paralegal","litigation","contract law","corporate law",
        "legal research","compliance","court filing","case management",
        "intellectual property"
    ],
    "Healthcare": [
        "patient care","nurse","clinical","medical assistant","healthcare",
        "treatment","therapy","rehabilitation","physician","medication",
        "public health","diagnosis"
    ],
    "HR & Operations": [
        "human resources","hr","recruitment","recruiter","talent acquisition","payroll",
        "onboarding","employee relations","benefits administration",
        "performance review","training development","workforce planning",
        "operations management","logistics","inventory management","vendor management"
    ],
    "Creative & Design": [
        "graphic design","illustrator","photoshop","digital media","branding",
        "ui design","ux design","content creation","copywriting","animation",
        "video editing","social media","creative direction"
    ],
    "Education": [
        "teacher","professor","lecturer","curriculum development","lesson planning",
        "student engagement","tutoring","instruction","classroom management"
    ],
    "Manufacturing & Construction": [
        "construction","blueprint","mechanical","electrical","civil engineering",
        "safety compliance","production line","machinery","quality assurance"
    ],
    "Agriculture & Environment": [
        "agriculture","farming","crop","sustainability","ecology","forestry",
        "environmental management","conservation","water resources"
    ],
    "Hospitality & Food": [
        "chef","menu planning","hospitality","catering","food safety","sanitation",
        "guest relations","restaurant operations"
    ],
    "Other Services": [
        "customer service","client service","call center","retail","merchandising",
        "aviation","pilot","flight attendant","bpo"
    ]
}


# ============================================
# CLEAN TEXT FOR DOMAIN / KEYWORD MATCHING
# ============================================
def clean_text_for_domain(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for sw in custom_stopwords:
        text = text.replace(f" {sw} ", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================
# DOMAIN DETECTION
# ============================================
def detect_domain(text):
    text = text.lower()
    domain_scores = {d: 0 for d in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                domain_scores[domain] += 1
    best = max(domain_scores, key=domain_scores.get)
    return best, domain_scores


# ============================================
# DOMAIN SIMILARITY (cosine over domain tag counts)
# ============================================
def domain_similarity(job_dom_vec, res_dom_vec):
    jv = np.array(list(job_dom_vec.values()))
    rv = np.array(list(res_dom_vec.values()))
    denom = (np.linalg.norm(jv) * np.linalg.norm(rv)) + 1e-9
    return float(np.dot(jv, rv) / denom)


# ============================================
# SKILLS
# ============================================
skills = [
    "excel","word","powerpoint","sql","python","project management",
    "data analysis","ms office","microsoft office"
]

def skill_match(job_text, resume_text):
    job = job_text.lower()
    res = resume_text.lower()
    count = sum(1 for s in skills if s in job and s in res)
    return count / len(skills)


# ============================================
# EDUCATION LEVELS
# ============================================
education_levels = {
    "phd": 4,
    "master": 3,
    "bachelor": 2,
    "associate": 1,
    "diploma": 0.5
}

def education_score(text):
    text = text.lower()
    for level, score in education_levels.items():
        if level in text:
            return score
    return 0


# ============================================
# EXPERIENCE SENIORITY
# ============================================
experience_words = ["senior", "lead", "manager", "director", "specialist", "analyst"]

def seniority_score(text):
    text = text.lower()
    return sum(1 for w in experience_words if w in text)


# ============================================
# KEYWORD OVERLAP
# ============================================
def keyword_overlap(job_text, resume_text):
    j = set(job_text.lower().split())
    r = set(resume_text.lower().split())
    if not j or not r:
        return 0
    return len(j & r) / len(j | r)
e