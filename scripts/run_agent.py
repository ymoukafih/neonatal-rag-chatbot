"""
Automatically fetches and embeds ALL neonatal & pediatric topics into ChromaDB.
Usage: uv run python scripts/run_agent.py
"""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.logging_config import setup_logging
setup_logging("INFO")

from src.agents.pubmed_agent import pubmed_agent
from src.agents.state import AgentState

# ─── Complete Neonatal & Pediatric Knowledge Base ─────────────────────────────

NEONATAL_TOPICS = [

    # ── General Neonatal Care ─────────────────────────────────────────────────
    "neonatal care newborn assessment Apgar score",
    "neonatal vital signs normal ranges monitoring",
    "neonatal thermoregulation temperature management incubator",
    "kangaroo mother care skin to skin newborn",
    "neonatal skin care hygiene vernix",
    "neonatal weight gain growth charts",
    "neonatal discharge criteria follow-up",
    "neonatal circumcision care",
    "neonatal hearing screening OAE ABR",
    "newborn screening metabolic disorders bloodspot",

    # ── Nutrition & Feeding ───────────────────────────────────────────────────
    "neonatal breastfeeding initiation benefits",
    "neonatal formula feeding types preparation",
    "neonatal total parenteral nutrition TPN premature",
    "neonatal enteral feeding nasogastric tube",
    "neonatal donor human milk bank",
    "neonatal vitamin D iron supplementation",
    "neonatal lactation support breast milk expression",
    "neonatal swallowing feeding difficulties",
    "neonatal growth failure failure to thrive",

    # ── Respiratory ───────────────────────────────────────────────────────────
    "neonatal respiratory distress syndrome RDS surfactant",
    "neonatal transient tachypnea newborn TTN",
    "neonatal apnea prematurity caffeine treatment",
    "neonatal oxygen therapy monitoring SpO2",
    "neonatal CPAP continuous positive airway pressure",
    "neonatal mechanical ventilation high frequency",
    "neonatal surfactant replacement therapy INSURE",
    "bronchopulmonary dysplasia chronic lung disease premature",
    "neonatal meconium aspiration syndrome",
    "neonatal pneumothorax air leak syndrome",
    "neonatal pulmonary hemorrhage",
    "neonatal choanal atresia airway obstruction",
    "neonatal diaphragmatic hernia CDH",
    "neonatal tracheoesophageal fistula",

    # ── Cardiovascular ────────────────────────────────────────────────────────
    "neonatal congenital heart disease diagnosis",
    "patent ductus arteriosus PDA treatment indomethacin ibuprofen",
    "neonatal persistent pulmonary hypertension PPHN nitric oxide",
    "neonatal cardiac arrhythmia SVT treatment",
    "neonatal heart failure management",
    "neonatal hypoplastic left heart syndrome",
    "neonatal tetralogy of fallot",
    "neonatal transposition great arteries",
    "neonatal coarctation aorta",
    "neonatal echocardiography cardiac screening",
    "neonatal prostaglandin ductal dependent heart disease",

    # ── Neurological ─────────────────────────────────────────────────────────
    "neonatal hypoxic ischemic encephalopathy HIE cooling",
    "neonatal seizures phenobarbital levetiracetam treatment",
    "neonatal intraventricular hemorrhage IVH grading",
    "neonatal periventricular leukomalacia PVL",
    "neonatal brain MRI imaging",
    "neonatal amplitude integrated EEG aEEG monitoring",
    "neonatal hydrocephalus post hemorrhagic",
    "neonatal meningitis CSF diagnosis",
    "neonatal stroke perinatal arterial",
    "neonatal hypotonia floppy infant",
    "neonatal neural tube defects spina bifida",
    "neonatal microcephaly macrocephaly head circumference",

    # ── Infectious Disease ────────────────────────────────────────────────────
    "neonatal sepsis early onset GBS diagnosis",
    "neonatal sepsis late onset coagulase negative staph",
    "neonatal sepsis biomarkers CRP procalcitonin",
    "neonatal meningitis bacterial treatment lumbar puncture",
    "neonatal pneumonia congenital acquired",
    "neonatal herpes simplex virus HSV acyclovir",
    "neonatal group B streptococcus GBS prophylaxis",
    "congenital infections TORCH cytomegalovirus CMV",
    "neonatal congenital toxoplasmosis treatment",
    "neonatal congenital rubella syndrome",
    "neonatal congenital syphilis penicillin",
    "neonatal COVID-19 SARS-CoV-2",
    "neonatal candida fungal infection fluconazole",
    "neonatal MRSA antibiotic resistant infection",
    "neonatal antibiotic stewardship aminoglycoside",

    # ── Metabolic & Endocrine ─────────────────────────────────────────────────
    "neonatal jaundice hyperbilirubinemia phototherapy exchange",
    "neonatal hypoglycemia glucose monitoring dextrose",
    "neonatal hypocalcemia hypomagnesemia electrolytes",
    "neonatal metabolic acidosis base deficit",
    "neonatal hypernatremia hyponatremia sodium",
    "neonatal hyperkalemia potassium ECG",
    "neonatal congenital hypothyroidism screening",
    "neonatal congenital adrenal hyperplasia CAH",
    "neonatal diabetes mellitus transient permanent",
    "neonatal inborn errors metabolism organic acidemia",
    "neonatal phenylketonuria PKU screening",
    "neonatal galactosemia diagnosis treatment",
    "neonatal maple syrup urine disease MSUD",
    "neonatal hyperbilirubinemia Rh ABO incompatibility",

    # ── Gastrointestinal ──────────────────────────────────────────────────────
    "necrotizing enterocolitis NEC premature prevention",
    "neonatal bowel obstruction malrotation volvulus",
    "neonatal gastroesophageal reflux GERD",
    "neonatal abdominal wall defects gastroschisis omphalocele",
    "neonatal Hirschsprung disease constipation",
    "neonatal intestinal atresia duodenal jejunal",
    "neonatal imperforate anus anorectal malformation",
    "neonatal biliary atresia jaundice cholestasis",
    "neonatal short bowel syndrome intestinal failure",
    "neonatal meconium ileus cystic fibrosis",

    # ── Hematology ────────────────────────────────────────────────────────────
    "neonatal anemia hemoglobin erythropoietin",
    "neonatal polycythemia hyperviscosity",
    "neonatal thrombocytopenia platelet transfusion",
    "neonatal coagulopathy DIC vitamin K",
    "hemolytic disease newborn Rh ABO incompatibility",
    "neonatal blood transfusion guidelines threshold",
    "neonatal sickle cell disease screening",
    "neonatal neutropenia infection risk",

    # ── Prematurity ───────────────────────────────────────────────────────────
    "extremely premature infant less than 28 weeks outcomes",
    "very low birth weight VLBW infant care",
    "late preterm infant 34 36 weeks complications",
    "small for gestational age SGA growth restriction",
    "large for gestational age LGA macrosomia",
    "retinopathy of prematurity ROP screening laser",
    "neonatal osteopenia prematurity bone disease",
    "neonatal inguinal hernia premature repair",

    # ── Procedures & Interventions ────────────────────────────────────────────
    "neonatal resuscitation NRP guidelines algorithm",
    "neonatal umbilical venous arterial catheter",
    "neonatal PICC line peripherally inserted catheter",
    "neonatal intubation laryngoscopy endotracheal tube",
    "neonatal chest tube pleural drainage",
    "neonatal lumbar puncture technique",
    "neonatal exchange transfusion jaundice",
    "neonatal phototherapy LED double",
    "neonatal therapeutic hypothermia cooling blanket",
    "neonatal ECMO extracorporeal membrane oxygenation",
    "neonatal surgical consult indications",

    # ── Pharmacology ─────────────────────────────────────────────────────────
    "neonatal antibiotic dosing ampicillin gentamicin",
    "neonatal pain management morphine fentanyl sucrose",
    "neonatal caffeine apnea loading dose",
    "neonatal sedation midazolam phenobarbital",
    "neonatal drug withdrawal neonatal abstinence syndrome NAS",
    "neonatal ibuprofen indomethacin PDA closure",
    "neonatal steroids dexamethasone hydrocortisone",
    "neonatal diuretics furosemide",
    "neonatal antifungal fluconazole amphotericin",
    "neonatal vasopressors dopamine dobutamine",
    "neonatal drug pharmacokinetics dosing adjustment",

    # ── Special Populations ───────────────────────────────────────────────────
    "infant of diabetic mother IDM hypoglycemia",
    "neonatal drug exposure opioid NAS scoring Finnegan",
    "twin pregnancy neonatal outcomes discordance",
    "neonatal twins transfusion TTTS syndrome",
    "neonatal post cardiac surgery care",
    "neonatal transport stabilization outborn",

    # ── Developmental & Long-term ─────────────────────────────────────────────
    "neonatal neurodevelopmental outcomes follow-up NICU",
    "neonatal developmental care positioning NIDCAP",
    "neonatal palliative care end of life comfort",
    "neonatal family centered care parents NICU",
    "neonatal pain assessment NIPS PIPP scale",
    "neonatal brain development white matter",
    "premature infant cognitive motor outcome school age",
    "NICU graduate developmental follow-up program",

    # ── Pediatric General ─────────────────────────────────────────────────────
    "pediatric growth development milestones",
    "pediatric fever management antipyretic",
    "pediatric dehydration fluid replacement oral rehydration",
    "pediatric respiratory infection bronchiolitis RSV",
    "pediatric pneumonia community acquired treatment",
    "pediatric urinary tract infection diagnosis antibiotics",
    "pediatric asthma management inhaler corticosteroid",
    "pediatric anemia iron deficiency treatment",
    "pediatric failure to thrive nutrition assessment",
    "pediatric vaccination immunization schedule",
    "pediatric seizure febrile convulsion management",
    "pediatric meningitis bacterial lumbar puncture",
    "pediatric gastroenteritis rotavirus norovirus",
    "pediatric constipation management polyethylene glycol",
    "pediatric allergies food allergy anaphylaxis",
    "pediatric eczema atopic dermatitis treatment",
    "pediatric growth hormone deficiency treatment",
    "pediatric type 1 diabetes insulin management",
    "pediatric obesity metabolic syndrome",
    "pediatric developmental delay autism spectrum",
]


def run_topic(topic: str) -> None:
    """Run the PubMed agent for a single topic."""
    initial_state = AgentState(topic=topic)
    final_state = pubmed_agent.invoke(initial_state)
    print(final_state["summary"])
    print("─" * 60)


if __name__ == "__main__":
    total = len(NEONATAL_TOPICS)
    print(f"\n🩺 Neonatal & Pediatric Knowledge Base Builder")
    print(f"📚 Fetching {total} topics from PubMed...\n")
    print("─" * 60)

    success = 0
    failed  = 0

    for i, topic in enumerate(NEONATAL_TOPICS, 1):
        print(f"\n[{i}/{total}] {topic}")
        try:
            run_topic(topic)
            success += 1
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed += 1

    print(f"\n{'═' * 60}")
    print(f"✅ Completed : {success}/{total} topics ingested")
    if failed:
        print(f"❌ Failed    : {failed} topics")
    print(f"📊 Your ChromaDB now contains comprehensive neonatal")
    print(f"   and pediatric knowledge ready for clinical Q&A.")
    print(f"{'═' * 60}\n")