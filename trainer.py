import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.special import softmax

# Ensure accelerate is installed
try:
    import accelerate
except ImportError:
    raise ImportError("Please install accelerate: pip install 'accelerate>=0.26.0'")

# Load Data
data = [
    {
        "case_id": "case_001",
        "case_title": "Dispute Between Home Buyer and Promoter for Project MY CITY - PHASE II - CLUSTER 4",
        "case_description": "A home buyer disputes the terms of the agreement for sale presented by the promoter, citing ambiguities and contradictions with the cost sheet provided at booking. The buyer seeks execution of an amended agreement and compensation for delays.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Home Buyer’s Claims",
            "section_description": "The complainant seeks several reliefs, including an amended agreement for sale, possession of the apartment, and compensation for mental harassment."
          },
          {
            "section_id": "section_002",
            "section_title": "Promoter’s Defense",
            "section_description": "The promoter claims the draft agreement aligns with other allottees and contends that only 4% of the payment has been made by the complainant, which does not meet the minimum requirement under RERA."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "The Authority observed that there was no agreement for sale and the complainant had not demonstrated sufficient locus standi due to unsubstantiated allegations and minimal payment towards the property."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "The Authority dismissed the complaint, citing lack of proper documentation and unsubstantiated allegations, and imposed a penalty of Rs.10,000/- on the complainant."
          }
        ],
        "strong_points": [
          "Complainant raised concerns about ambiguities in the draft agreement and discrepancies from the cost sheet.",
          "Complainant’s willingness to pay the remaining amount and proceed with the project."
        ],
        "weak_points": [
          "Complainant has paid only 4percent of the total consideration, falling short of the RERA requirement.",
          "No signed agreement for sale exists to substantiate claims."
        ]
      },
      {
        "case_id": "case_002",
        "case_title": "Complaint No. CC006000000480080 - Sachin Babu Gurav vs Damodar Suruchi Developers Pvt. Ltd.",
        "case_description": "Sachin Babu Gurav filed a complaint seeking directions from MahaRERA to hand over possession of flat no. 108 in Wing B of the project 'UK Iridium', along with interest and compensation for delayed possession. The matter was referred to MahaRERA Conciliation Forum, where both parties resolved the issue amicably. The complaint was subsequently withdrawn following the signed consent terms.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Request",
            "section_description": "The complainant sought directions from MahaRERA for the respondent to hand over possession of flat no. 108 in Wing B of the registered project 'UK Iridium', and also to pay interest and compensation for delayed possession under section 18 of RERA."
          },
          {
            "section_id": "section_002",
            "section_title": "Conciliation Attempt",
            "section_description": "Both parties expressed willingness to resolve the matter amicably and requested the complaint be referred to MahaRERA Conciliation Forum for further action."
          },
          {
            "section_id": "section_003",
            "section_title": "Conciliation Outcome",
            "section_description": "The matter was resolved amicably in the MahaRERA Conciliation Forum, and the parties signed consent terms on 07-06-2024. The complaint was then transferred back to the bench."
          },
          {
            "section_id": "section_004",
            "section_title": "Complaint Withdrawal",
            "section_description": "The complaint was scheduled for hearing again, but the complainant did not appear. Since the matter was settled and the consent terms signed, the withdrawal application was accepted, and the complaint was disposed of as settled and withdrawn."
          }
        ],
        "strong_points": [
          "Both parties agreed to resolve the matter amicably through the MahaRERA Conciliation Forum.",
          "Consent terms were signed and the matter was officially recorded as resolved."
        ],
        "weak_points": [
          "The complainant did not appear at the final hearing nor submitted any objections to the consent terms or withdrawal application."
        ]
      },
      {
        "case_id": "case_003",
        "case_title": "Complaint No. CC005000000258951 - Shirish Govind Sukhatme vs Om Developer",
        "case_description": "Shirish Govind Sukhatme filed a complaint seeking directions from MahaRERA for the respondent to show 12 flats booked by him as sold inventory in the project 'Tropica Phase II'. The complaint was withdrawn after MahaRERA confirmed that the flats were reflected as sold on the website.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Request",
            "section_description": "The complainant sought directions from MahaRERA to the respondent to show 12 flats he booked as sold inventory in the project 'Tropica Phase II', registered with MahaRERA under no. P52100008711."
          },
          {
            "section_id": "section_002",
            "section_title": "MahaRERA Review",
            "section_description": "MahaRERA reviewed the complaint and found that the flats were correctly listed as sold inventory on the MahaRERA website. The complainant was informed of this and verified the information."
          },
          {
            "section_id": "section_003",
            "section_title": "Complaint Withdrawal",
            "section_description": "After confirming the issue was resolved, the complainant filed a withdrawal application, which was accepted, and the complaint was disposed of."
          }
        ],
        "strong_points": [
          "The complainant verified that the flats were listed as sold inventory on the MahaRERA website after MahaRERA's review.",
          "The complaint was resolved amicably without further legal action."
        ],
        "weak_points": [
          "The complainant initially believed the flats were not listed as sold inventory, leading to the complaint."
        ]
      },
      {
        "case_id": "case_004",
        "case_title": "Complaint No. CC006000000055076 - Mr. Shekhar Bhargav Devlekar vs M/s. Nirmal Lifestyle (Kalyan) Pvt. Limited",
        "case_description": "Shekhar Bhargav Devlekar filed a complaint seeking compensation for delayed possession of a flat. The complaint was withdrawn after both parties signed consent terms, but a non-execution application was later filed due to non-compliance with the terms.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Request",
            "section_description": "The complainant sought compensation for delayed possession under section 18 of RERA for flat no. D-308 in the project 'Lifestyle City Kalyan'."
          },
          {
            "section_id": "section_002",
            "section_title": "Consent Terms",
            "section_description": "Both parties filed consent terms on 11th October 2018, settling the matter amicably. The complaint was withdrawn on 15th October 2018 after the consent terms were filed."
          },
          {
            "section_id": "section_003",
            "section_title": "Non-Execution Application",
            "section_description": "The complainant filed a non-execution application in 2020 due to non-compliance with the consent terms, claiming the respondent had not paid the agreed interest. MahaRERA acknowledged the issue but stated no further action could be taken due to the ongoing insolvency proceedings of the respondent."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "The MahaRERA disposed of the non-execution application, stating that due to the insolvency proceedings of the respondent, no further action could be taken, but the complainant was advised to file a claim with the Resolution Professional."
          }
        ],
        "strong_points": [
          "The complaint was resolved through consent terms, and both parties agreed to settle amicably.",
          "The complainant took proactive steps to address non-compliance."
        ],
        "weak_points": [
          "The respondent failed to comply with the consent terms, leading to the non-execution application."
        ]
      },
      {
        "case_id": "case_005",
        "case_title": "Complaint No. CC006000000192449 - Mr. Hamendra Kasturlal Doshi vs M/s. Rite Developers Pvt. Ltd.",
        "case_description": "Mr. Hamendra Kasturlal Doshi filed a complaint seeking interest for delayed possession of flat no. 1504 in the project 'Rite Skyluxe,' as per the agreement for sale dated 29/06/2017. Although possession was delayed beyond the agreed date of 31/03/2018, the flat was eventually handed over on 07/01/2020. The complainant alleges contravention of the agreement and seeks compensation under Section 18 of RERA.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant seeks interest for delayed possession of flat no. 1504 from 31/03/2018 to 07/01/2020, as per the provisions of Section 18 of RERA. The complainant asserts that the possession letter signed at the time of handover does not override legal provisions."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent’s Defense",
            "section_description": "The respondent argues that the complaint is not maintainable since it was filed after possession was taken, and the complainant had waived all claims by signing the possession letter. The respondent attributes the delay to changes in government policy affecting approvals and requests dismissal of the complaint."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the respondent delayed possession beyond the agreed date without adequate justification. Section 18(3) of RERA allows compensation when possession is delayed contrary to the agreement. The Authority also noted that the quantum of compensation must be determined by the Adjudicating Officer under Section 71 of RERA."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the case to the Adjudicating Officer for a decision on the quantum of compensation, acknowledging the complainant's right to relief under Section 18(3) of RERA."
          }
        ],
        "strong_points": [
          "The complainant paid the full consideration as per the agreement and sought interest for delay within legal provisions.",
          "MahaRERA recognized the applicability of Section 18(3), which entitles the complainant to compensation for delayed possession."
        ],
        "weak_points": [
          "The respondent cited changes in government policy as a factor beyond their control, leading to delays.",
          "The complainant signed the possession letter, which the respondent argues constitutes a waiver of claims."
        ]
      },
      {
        "case_id": "case_006",
        "case_title": "Complaint No. CC006000000057281 - Mr. Anil Gulabdas Shah vs M/s. Solaris Developers Private Limited & Anr.",
        "case_description": "Mr. Anil Gulabdas Shah filed a complaint seeking a refund of the full amount paid for flat no. 1601 in the project 'Solaris Titanium Tower' under Section 18 of RERA, citing non-receipt of an occupancy certificate. Although the complainant had taken possession of the flat in 2013, the MahaRERA observed that the promoter failed to obtain the occupancy certificate and referred the case for compensation under Section 18(3) of RERA.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant sought a refund of the full amount paid towards the consideration of the flat, along with interest and compensation, citing violations under Section 18 of RERA. He also filed an application to transfer the complaint to the Adjudicating Officer to decide compensation."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "Respondent No. 1 informed MahaRERA that they had already applied for an extension of the project validity period. Respondent No. 2, the society, claimed that they were not a necessary party to the complaint and requested their name be deleted."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the complainant had taken possession of the flat in 2013, and his request for a full refund was not justified as he had occupied the flat for over seven years. However, the Authority noted that the promoter had failed to obtain the occupancy certificate and recognized the complainant’s right to seek compensation for this lapse under Section 18(3) of RERA."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA allowed the complainant’s application to transfer the complaint to the Adjudicating Officer for determining the quantum of compensation under Section 18(3) of RERA. The case was transferred for a suitable decision as per the rules and regulations."
          }
        ],
        "strong_points": [
          "The complainant emphasized the lack of an occupancy certificate for the flat, which is a significant lapse by the promoter.",
          "MahaRERA acknowledged the applicability of Section 18(3) of RERA, entitling the complainant to seek compensation for the promoter’s failure."
        ],
        "weak_points": [
          "The complainant had already taken possession of the flat in 2013, which weakened his case for a full refund.",
          "The respondent cited their application for an extension of the project validity period, suggesting ongoing compliance efforts."
        ]
      },
      {
        "case_id": "case_007",
        "case_title": "Complaint No. CC006000000055459 - M/s. Nisar Properties Pvt. Ltd. vs Mr. Shaikh Haroon Rashid & Mr. Shaikh Abuzar Haroon Rashid",
        "case_description": "M/s. Nisar Properties Pvt. Ltd. filed a complaint seeking payment of outstanding dues along with interest under Section 19(6) of RERA for a flat in the project 'Rajal-Om Jaishriram CHS Ltd.' The complainant also sought compensation of Rs. 10,00,000/- for mental agony caused by the respondents' alleged violations of RERA. Although possession was handed over in January 2018, the complainant stated it was done without prejudice to their claims for dues and compensation.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant sought recovery of outstanding dues along with interest as per Section 19(6) of RERA. Additionally, the complainant demanded compensation of Rs. 10,00,000/- for mental harassment caused by the respondents' failure to pay the dues despite taking possession of the flat."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondents acknowledged taking possession of the flat in January 2018 but disputed the claims for outstanding dues and compensation. They did not provide any written submissions challenging the claims during the hearing."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the respondents had taken possession of the flat, and the complainant alleged violations of Section 19(6) due to non-payment of dues. The Authority noted that Sections 71 and 72 of RERA empower the Adjudicating Officer to decide on the quantum of compensation for violations."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the case to the Adjudicating Officer for a suitable decision on the quantum of compensation under Sections 71 and 72 of RERA. The Authority directed the parties to present their arguments before the Adjudicating Officer."
          }
        ],
        "strong_points": [
          "The complainant provided evidence of outstanding dues as per the agreement for sale.",
          "MahaRERA acknowledged the applicability of Sections 71 and 72, empowering the Adjudicating Officer to decide on compensation."
        ],
        "weak_points": [
          "The respondents disputed the claims but did not provide a strong defense or written submissions to challenge the complainant’s allegations.",
          "The complainant had already handed over possession of the flat, potentially weakening their position regarding claims of harassment."
        ]
      },
      {
        "case_id": "case_008",
        "case_title": "Complaint No. CC006000000292629 - Pradip Shripat Patankar vs M/S. Sheltrex Karjat Pvt Ltd",
        "case_description": "Pradip Shripat Patankar filed a complaint seeking a refund of the full booking amount paid for flat no. 1603 in Building Y10 of the project 'Sheltrex Smart Phone City Project 1 - Phase II,' along with interest under the provisions of RERA. However, the respondent has been admitted into Corporate Insolvency Resolution Process (CIRP) as per the National Company Law Tribunal (NCLT) order, leading to a moratorium under Section 14 of the Insolvency and Bankruptcy Code, 2016. Consequently, MahaRERA has adjourned the complaint sine die.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant sought a refund of the full booking amount paid towards flat no. 1603 in Building Y10 of the respondent’s project, along with interest, citing violations under RERA. He expressed concerns over the lack of progress in the project and demanded relief from MahaRERA."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent’s Status",
            "section_description": "MahaRERA noted that the respondent has been admitted into CIRP as per the NCLT order dated 01-02-2023 in CP (IB) No. 3126/MB/2019. This invokes a moratorium under Section 14 of the Insolvency and Bankruptcy Code, prohibiting the continuation of legal proceedings against the respondent."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA acknowledged that legal proceedings against the respondent are barred under Section 14 of the Insolvency and Bankruptcy Code due to the ongoing CIRP. The complainant informed MahaRERA that he had already filed his claim with the appointed Insolvency Resolution Professional (IRP)."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA adjourned the complaint sine die, citing the moratorium under Section 14 of the Insolvency and Bankruptcy Code. The complainant was granted the liberty to approach MahaRERA again after the final disposal of the insolvency proceedings, if necessary."
          }
        ],
        "strong_points": [
          "The complainant has already taken proactive steps by filing a claim with the appointed IRP under the CIRP.",
          "MahaRERA provided clarity on the legal constraints due to the moratorium under Section 14 of the Insolvency and Bankruptcy Code."
        ],
        "weak_points": [
          "The moratorium prevents MahaRERA from taking any immediate action or providing relief to the complainant.",
          "The respondent’s insolvency proceedings could significantly delay the complainant’s ability to recover the booking amount or seek compensation."
        ]
      },
      {
        "case_id": "case_009",
        "case_title": "Complaint No. CC005000000043784 - Mr. Umesh Sandipan Magar & Mts. Jyoti Umesh Magar vs M/s. Kul Developers Pvt Ltd. & M/s. Ashdan Developers Private Ltd.",
        "case_description": "The complainants filed a complaint seeking a refund of the amount paid for flat no. 1103 in the project 'VTP Beaumonde-Nation R2 Tower 5,' along with interest and compensation under Section 18 of the RERA. The respondents failed to hand over possession of the flat on the agreed date of possession as per the registered agreement for sale. The case was transferred to the Adjudicating Officer of MahaRERA for determination of the compensation and interest under Section 18 of the RERA.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants' Claims",
            "section_description": "The complainants alleged that they booked the flat in 2013 based on the goodwill of Kumar Builders, and a registered agreement for sale was executed on 11-12-2014. As per the agreement, the possession was to be handed over by 11-12-2019, which the respondents failed to do. They sought a refund, interest, and compensation for the delay under Section 18 of the RERA."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents' Position",
            "section_description": "The respondents denied the claims, citing a change in shareholding in 2018 and stating that the project is now 80-85% complete, with a revised completion date supported by other allottees. However, no plausible explanation was provided for the delay, and no written reply was filed."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the respondent violated Section 18 of the RERA by failing to hand over possession by the agreed date. The agreement for sale and other evidence supported the complainants' claims. The respondent did not provide any valid reasons for the delay and failed to comply with an earlier MahaRERA order issued in 2018 directing possession by 11-12-2019."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA upheld the complainants' right to a refund, interest, and compensation under Section 18 of the RERA. However, determining the quantum of compensation was beyond MahaRERA’s jurisdiction and was referred to the Adjudicating Officer for further decision."
          }
        ],
        "strong_points": [
          "The complainants provided a registered agreement for sale and demonstrated non-compliance by the respondent with the agreed possession date.",
          "MahaRERA recognized the respondents' failure to explain the delay and their previous non-compliance with an order issued in 2018."
        ],
        "weak_points": [
          "The case was transferred to the Adjudicating Officer for determining the compensation, potentially delaying relief for the complainants.",
          "The respondents have not clarified the revised completion timeline for the project, which could impact the complainants' claim."
        ]
      },
      {
        "case_id": "case_010",
        "case_title": "Complaint No. CC006000000198586 - M/s Aditya C HS Ltd vs M/s Navkar Builders and Developers",
        "case_description": "The complainant filed a complaint seeking directions from MahaRERA for the respondent to fulfill statutory and contractual obligations, complete the project, and compensate the society of allottees for expenses. The complaint also addressed the delay in project completion and possession of flats, citing violations under Sections 11, 18, and 19(3) of the RERA.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Claims",
            "section_description": "The complainant (a registered society of allottees) alleges that the respondent promoter has failed to fulfill statutory duties, resulting in project delays and financial losses for the society. The complainant seeks possession of flats, compensation for delay, and completion of the project under the provisions of RERA."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Position",
            "section_description": "The respondent claims the project is at a standstill, with 12 out of 21 floors constructed, and delays due to changes in MCGM norms. They argue possession has been handed over for some floors, but further construction is awaiting approvals. They also state the project is not complete due to regulatory issues."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA acknowledged the failure of the respondent to comply with RERA requirements, including the delay in completion and non-compliance with the agreed timeline. The authority also noted the respondent had failed to file a specific reply on maintainability and issues raised during hearings."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA held the complaint to be maintainable under Sections 11 and 14 of RERA and directed further proceedings. The complaint was adjourned for the next hearing to address the issues of maintainability and the responsibilities of the respondent under RERA."
          }
        ],
        "strong_points": [
          "The complainant presented a detailed account of delays and non-compliance by the respondent, including the failure to hand over possession and the absence of updates on the project’s progress.",
          "MahaRERA acknowledged the statutory violations and the need for further hearings to resolve the issues."
        ],
        "weak_points": [
          "The respondent failed to provide a clear explanation for the delay in construction and possession, and their reply lacked specifics on some key issues raised during the hearings.",
          "The case was adjourned for further proceedings, which may delay the final resolution of the complainant’s claims."
        ]
      },
      {
        "case_id": "case_011",
        "case_title": "Complaint No. CC006000000171668 - Smitesh Shivaji Patil vs Hubtown Limited",
        "case_description": "The complainant has filed a complaint regarding the delay in possession of an apartment purchased from the respondent’s project, ‘Hubtown Greenwoods D Wing.’ The complainant seeks interest on the delay and possession of the apartment, citing failure to meet the stipulated possession date as per the agreement for sale.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Claims",
            "section_description": "The complainant, Smitesh Shivaji Patil, alleges that the respondent, Hubtown Limited, has failed to deliver possession of the apartment (No. 405-D) within the stipulated period as per the agreement. The complainant seeks interest on the delay and requests the respondent to hand over possession of the apartment at the earliest."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Position",
            "section_description": "The respondent, Hubtown Limited, explains that construction delays were due to circumstances beyond their control. They offer the complainant the option of a refund with interest but note that the complainant has opted to stay in the project and await possession. The respondent has applied for the occupancy certificate (OC) but has not yet received it."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA acknowledged the complainant’s request for possession and the delay in the respondent’s completion of the project. The authority also recognized that the respondent has applied for the occupancy certificate but is unable to provide a definitive timeline for completion."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA directed the respondent to complete the project and obtain the occupancy certificate as per the revised timeline. The computation of interest on delay will be addressed in the final order, once the project is completed and the occupancy certificate is obtained."
          }
        ],
        "strong_points": [
          "The complainant presented a clear case of non-compliance with the possession date, and MahaRERA acknowledged the delay.",
          "MahaRERA has provided clear directions for the completion of the project and handling the delay once the occupancy certificate is obtained."
        ],
        "weak_points": [
          "The respondent failed to provide a specific timeline for the completion of the project or when possession could be handed over.",
          "The case was deferred to a final order, which may delay the resolution for the complainant."
        ]
      },
      {
        "case_id": "case_012",
        "case_title": "Complaint Nos. CC006000000078474, CC006000000078478, CC006000000078481, CC006000000079205, CC006000000089928 - Vinay Arvind Ranade & Others vs The Bombay Dyeing and Manufacturing Company Limited",
        "case_description": "The complainants filed complaints against The Bombay Dyeing and Manufacturing Company Limited, alleging failure to execute agreements for sale, and non-compliance with promised project timelines. The complainants seek a refund with interest and compensation, citing issues with the agreement for sale and non-completion of the project as promised. The dispute revolves around the delayed possession and non-execution of sale agreements despite the project nearing completion.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants' Claims",
            "section_description": "The complainants, who had booked apartments in the 'ICC' project by The Bombay Dyeing and Manufacturing Company Limited in 2012-2013, allege that the respondent failed to execute the agreement for sale and delayed possession. The draft agreement for sale is also claimed to be non-compliant with the Real Estate (Regulation and Development) Act, 2016. The complainants request a refund with interest and compensation for the delays and issues with the project."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Position",
            "section_description": "The respondent, The Bombay Dyeing and Manufacturing Company Limited, argues that the project was developed as per the sanctioned plans and approvals. The respondent claims that they have obtained the occupancy certificate (OC) in 2019 and are willing to execute the agreement for sale in compliance with the Real Estate Act. The respondent also expressed willingness to settle the dispute amicably."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that while the project was nearing completion, the complainants were entitled to a remedy under Section 12 of the Real Estate Act, which mandates a refund due to non-compliance. The authority considered the larger impact on the project, noting that allowing bulk withdrawals by multiple complainants would jeopardize the completion of the project."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the complaints to the Adjudicating Officer (AO) for adjudication of interest and compensation under the relevant provisions of the Real Estate Act. The decision followed the ruling of the Maharashtra Real Estate Appellate Tribunal, which held that the complainants were entitled to a remedy under Section 12 of the Act."
          }
        ],
        "strong_points": [
          "MahaRERA acknowledged the complainants' entitlement to a refund under Section 12 of the Real Estate Act and transferred the complaints to the Adjudicating Officer for further adjudication.",
          "The project was nearing completion, and the respondent had obtained the occupancy certificate, indicating progress on the project."
        ],
        "weak_points": [
          "The respondent's failure to execute the agreements for sale and provide timely possession led to the complaints.",
          "The project nearing completion did not prevent the complainants' legitimate claims under the Real Estate Act, potentially delaying the final resolution."
        ]
      },
      {
        "case_id": "case_013",
        "case_title": "Complaint No. CC005000000053927 - Mr. Swapnil Shahaji Patil vs M/s. Maple Buildcon",
        "case_description": "The complainant filed a complaint seeking a refund of the amount paid for a flat booked in the respondent's project, 'Aapla Ghar Ambegaon Annex-Phase 1', due to the failure to hand over possession of the flat as per the agreement. The complainant paid a substantial amount but no possession was given, and the project was reportedly at a standstill. The complainant seeks a refund under section 18 of the RERA Act.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Claims",
            "section_description": "The complainant booked flat No. 609 in the 'Aapla Ghar Ambegaon Annex-Phase 1' project for Rs. 18,44,812/-, paying Rs. 6,13,192/- as part of the total consideration. Despite a registered agreement for sale, the possession of the flat was not given by the agreed date of 31-12-2019. The complainant seeks a refund with interest and compensation under section 18 of the RERA Act due to the project being at a standstill."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Position",
            "section_description": "The respondent argues that the project completion date is 31-12-2022 and asserts that the flat is ready for fit-out possession. The respondent also claims that the completion certificate will be obtained before the end of 2022. They argue that the complaint is premature and request the dismissal of the case."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority's Observations",
            "section_description": "MahaRERA found that the agreed date of possession had lapsed and the respondent could not provide sufficient justification for the delay. Since the complainant is seeking a refund under section 18 of the RERA Act due to non-compliance with the agreed possession date, the case was transferred to the Adjudicating Officer for further decision."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the case to the Adjudicating Officer (AO) for a suitable decision regarding the refund and compensation under section 18 of the RERA Act. The AO is requested to consider the complainant's previous proceedings while deciding the case."
          }
        ],
        "strong_points": [
          "The complainant's claim for a refund is supported by the lapsed possession date in the agreement for sale.",
          "The transfer of the case to the Adjudicating Officer ensures that the complaint will be adjudicated in accordance with RERA provisions."
        ],
        "weak_points": [
          "The respondent's claim that the complaint is premature, as the project completion is scheduled for 31-12-2022, was not sufficient to dismiss the complaint."
        ]
      },
      {
        "case_id": "case_014",
        "case_title": "Complaint Nos. CC006000000078311 to CC006000000192983 - Various Complainants vs Ricardo Constructions Private Limited",
        "case_description": "Multiple complainants, including individuals and the US OPEN (ACES/MATCH POINT) AOA, filed complaints against Ricardo Constructions Pvt. Ltd. regarding delays in the project 'Ace and Matchpoint' at Mulund, Mumbai. The complainants seek refunds, possession of flats, or the execution of agreements for sale. The case involves the transfer of the project to a new promoter under the provisions of Section 15 of the RERA Act.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants' Claims",
            "section_description": "The complainants consist of allottees who purchased or booked apartments in the 'Ace and Matchpoint' project, seeking either refunds or possession of flats. The US OPEN AOA, representing over 200 allottees, has requested the respondent to start construction, provide a project completion schedule, and form a society, among other demands."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Position",
            "section_description": "The respondent, Ricardo Constructions Pvt. Ltd., argues that the project was transferred to them in August 2019 under the Securitisation and Reconstruction of Financial Assets and Enforcement of Securities Interest Act, 2002. They acknowledge the issues but are seeking to resolve them with the association and the allottees."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority's Observations",
            "section_description": "MahaRERA observed that the project had been transferred to the current promoter, and to protect the rights of the allottees, a transparent process should be initiated. It was decided that the US OPEN (ACES/MATCH POINT) AOA, with over 50% of the allottees as members, should play a crucial role in overseeing the project's progress."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA issued a series of directions, including the delivery of the list of allottees to the US OPEN (ACES/MATCH POINT) AOA, monthly meetings to discuss project progress, and the formation of an exit strategy for those wishing to withdraw. The respondent was ordered to execute agreements for sale within 60 days for those who have paid more than 10% of the consideration."
          }
        ],
        "strong_points": [
          "MahaRERA's decision ensures that the allottees' interests are protected, with a transparent approach for monitoring project progress.",
          "The decision includes specific timelines for actions, such as the execution of agreements for sale and initiation of society formation."
        ],
        "weak_points": [
          "The respondent may face challenges in meeting the strict timelines set by MahaRERA, especially considering the ongoing project delays."
        ]
      },
      {
        "case_id": "case_015",
        "case_title": "Complaint No. CC006000000161244 - Avinash Dudumkar & Devyani Dudumkar vs Safal Realtors And Developers Pvt Ltd & Ors.",
        "case_description": "The complainants, Avinash Dudumkar and Devyani Dudumkar, filed a complaint against Safal Realtors And Developers Pvt. Ltd. for non-execution of a registered agreement for sale and failure to hand over possession of a flat (No. 1001) in the project '22 Dhuleva'. They seek a refund of the money paid, along with interest and compensation under Section 18 of the RERA.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants' Claims",
            "section_description": "The complainants booked flat No. 1001 in the '22 Dhuleva' project for Rs. 1,45,00,000 and paid Rs. 21,75,000. Despite this, the respondents failed to execute a registered agreement for sale or hand over possession of the flat. The complainants seek a refund, along with interest and compensation as per Section 18 of the RERA."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents' Position",
            "section_description": "None of the respondents appeared for the hearing, and no defense was presented on their behalf during the proceedings."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority's Observations",
            "section_description": "MahaRERA observed that the complainants had filed for a refund and compensation under Section 18 of the RERA for acts of omission and commission by the respondents. The provision under Section 71 of RERA empowers the Adjudicating Officer to adjudge the quantum of compensation and interest."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the case to the Adjudicating Officer for a decision on the quantum of compensation under Section 18 of the RERA, as per the powers granted in Section 71."
          }
        ],
        "strong_points": [
          "The case demonstrates the mechanism of transferring complaints to the Adjudicating Officer for compensation decisions under RERA.",
          "MahaRERA is proactive in addressing omissions and ensuring that the compensation process is handled correctly."
        ],
        "weak_points": [
          "The lack of a defense from the respondents leaves the complainants' claims largely unchallenged, but the case still needs adjudication for compensation details."
        ]
      },
      {
        "case_id": "case_016",
        "case_title": "Complaint No. CC006000000192814 - Karan Chopra vs Skystar Buildcon Pvt Ltd",
        "case_description": "Karan Chopra filed a complaint against Skystar Buildcon Pvt Ltd seeking directions to obtain a full Occupancy Certificate, provide possession, and pay compensation for delayed possession of two flats (Nos. 603 and 604) in the project 'Sunteck City Avenue 1'. The complainant seeks relief under Sections 12, 14, 18, and 19 of the RERA for the delayed possession and misrepresentation about the possession date.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants' Claims",
            "section_description": "Karan Chopra seeks compensation under Sections 12 and 18 of RERA due to the respondent's failure to hand over possession of the flats (Nos. 603 and 604) on the agreed date mentioned in the agreement for sale. The complainant also highlights misrepresentation about the possession date, causing mental harassment."
          },
          {
            "section_id": "section_002",
            "section_title": "Conciliation Attempts",
            "section_description": "The complaint was referred to the Conciliation Forum for resolution. However, the parties could not reach an agreement, leading the forum to refer the case back to MahaRERA for further action."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority's Observations",
            "section_description": "MahaRERA noted that the complainant is entitled to seek compensation under Section 18(3) of RERA for the failure of the respondent to provide possession of the flats as per the terms of the agreement. The provisions of Section 71 of RERA empower the Adjudicating Officer to decide on the quantum of compensation."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "The case was transferred to the Adjudicating Officer, MahaRERA, Mumbai, for a suitable decision on compensation as per the provisions of RERA. The complainant was directed to serve a copy of the complaint to the respondent, who is allowed to file a reply."
          }
        ],
        "strong_points": [
          "MahaRERA effectively identified the complainant's entitlement to compensation under Section 18(3) of RERA.",
          "The case demonstrates the use of the Conciliation Forum before escalating to the Adjudicating Officer."
        ],
        "weak_points": [
          "The case has been delayed due to the failure to reach a conciliation agreement and the absence of the respondent's initial reply."
        ]
      },
      {
        "case_id": "case_017",
        "case_title": "Complaint No. CC006000000193280 - Renuka Sheth vs Era Realtors Pvt Ltd & Omkar Realtors & Developers Pvt Ltd",
        "case_description": "Renuka Sheth filed a complaint seeking a refund of the full amount paid for flat no. 1503 in the project 'Alta Monte and Signet' under Section 18 of RERA, citing the failure of the respondents to hand over possession of the flat within the agreed timeline. Despite a registered agreement, the respondents failed to deliver possession by March 31, 2020, and the project was delayed due to force majeure (COVID-19). The case was referred to the Adjudicating Officer for deciding the quantum of compensation under Section 18 of RERA.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant sought a refund of the amount paid for flat no. 1503, along with interest and compensation under Section 18 of RERA, due to the failure of the respondents to hand over possession of the flat by the agreed date. The complainant requested the transfer of the case to the Adjudicating Officer for deciding the compensation amount."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondents refuted the claim, explaining that the delay in handing over possession was due to the COVID-19 pandemic, which they argued should be considered a force majeure event when determining the compensation."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the complainant was entitled to relief under Section 18 of RERA due to the failure to deliver possession by the agreed date. The revised completion date for the project was set to June 30, 2024. The case was transferred to the Adjudicating Officer for a decision on the compensation quantum."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the complaint to the Adjudicating Officer to decide the quantum of compensation, in line with the provisions of Section 18 and Section 71 of RERA. The case was referred for further adjudication as per the rules."
          }
        ],
        "strong_points": [
          "The complainant's claim for a refund and compensation is backed by the failure of the respondents to deliver possession of the flat on time, as stipulated in the agreement.",
          "The case was appropriately transferred to the Adjudicating Officer for a fair decision on compensation under RERA."
        ],
        "weak_points": [
          "The respondents cited the COVID-19 pandemic as a force majeure event, which might weaken the complainant’s position regarding the timeline for possession.",
          "The revised completion date of June 30, 2024, introduces further delays, which may complicate the complainant's case for a full refund."
        ]
      },
      {
        "case_id": "case_018",
        "case_title": "Complaint No. CC005000000010700 - Mr. Shrant Gour & Dr. Purnima Gour vs Lake District Realty Private Limited & Pune Kondhwa Realty Private Limited",
        "case_description": "Mr. Shrant Gour and Dr. Purnima Gour filed a complaint seeking relief under Section 18 of RERA, requesting the handover of possession and a refund of the full consideration paid for flat no. 6602 in the project 'Cove - Lake District - Phase II'. The complainants sought compensation for the delayed possession. However, MahaRERA observed that the respondent, Lake District Realty Private Limited, was undergoing Corporate Insolvency Resolution Process (CIRP), as per NCLT proceedings initiated against them. As a result, the case was adjourned, with the complainants directed to file their claims before the CIRP, if any.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainants sought relief under Section 18 of RERA, requesting the handover of possession of flat no. 6602, along with a refund of the full consideration amount paid. They also sought compensation for the delay in possession and for the promoter’s failure to adhere to the timelines under the agreement."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondents did not specifically contest the claims regarding the delayed possession. However, it was revealed during the proceedings that the respondent company, Lake District Realty Private Limited, was under Corporate Insolvency Resolution Process (CIRP) initiated by NCLT, which led to the cessation of all legal proceedings against the company."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA acknowledged that the respondent was undergoing CIRP, and as per Section 14 of the Insolvency and Bankruptcy Code, all legal proceedings against the respondent were prohibited. Therefore, the authority could not proceed with the complaint at this stage, and the complainants were advised to file their claims with the CIRP process."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA adjourned the complaint sine die due to the respondent's involvement in CIRP, as per the provisions of the Insolvency and Bankruptcy Code. The complainants were granted the liberty to approach MahaRERA after the final disposal of the insolvency proceedings, if necessary."
          }
        ],
        "strong_points": [
          "The complainants were proactive in seeking relief under Section 18 of RERA, highlighting the delay in possession and the need for compensation.",
          "MahaRERA recognized the CIRP proceedings and appropriately adjourned the case in accordance with the Insolvency and Bankruptcy Code."
        ],
        "weak_points": [
          "The respondent's insolvency status prohibited the immediate resolution of the complaint, delaying the resolution process for the complainants.",
          "The complainants may face additional delays as the CIRP process takes precedence over RERA proceedings."
        ]
      },
      {
        "case_id": "case_019",
        "case_title": "Complaint No. CC005000000011054 - Vikas Pandit Kirange vs Naiknavare Housing Developments Pvt. Ltd.",
        "case_description": "Vikas Pandit Kirange filed a complaint seeking a refund of the amount paid for the 'Dwarka Project 2', along with rent compensation and interest due to delayed possession. The complainant had initially withdrawn the complaint in 2018 after a mutual settlement, but later filed for non-compliance of the agreed terms. MahaRERA transferred the case to the Secretary, MahaRERA for further action after unsuccessful conciliation attempts.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant sought the following reliefs: rent compensation for the delayed possession, interest on the total amount disbursed, waiver of GST on the remaining amount, compensation for filing the complaint, and a firm possession date confirmation from the builder."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondent stated that the online complaint was incomplete and the issues raised were unclear, seeking additional time to file a detailed reply. The respondent also noted discrepancies in the reliefs sought."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA noted that the complaint had been withdrawn in 2018 after a mutual settlement. However, the complainant had filed for non-compliance, and after unsuccessful conciliation, the matter was referred back to the authority. The case was transferred to the Secretary, MahaRERA for further action."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA observed that the complainant’s case regarding non-compliance of the 2018 settlement order had merit and transferred the case to the Secretary, MahaRERA for further action and resolution."
          }
        ],
        "strong_points": [
          "The complainant was proactive in filing for non-compliance after the settlement, ensuring that the issue of delayed possession was addressed.",
          "MahaRERA appropriately transferred the case to the Secretary for further action, considering the non-compliance and failed conciliation attempts."
        ],
        "weak_points": [
          "The initial settlement in 2018 and the complainant’s subsequent withdrawal of the case weakened their position.",
          "The respondent's request for more time and objections to the clarity of the complaint caused delays in the resolution of the matter."
        ]
      },
      {
        "case_id": "case_020",
        "case_title": "Complaint No. CC006000000000534 - Sohail Moosa vs Enkay Castle",
        "case_description": "Sohail Moosa filed a complaint seeking interest and compensation for delayed possession and non-provision of amenities in the project 'Enkay Garden - Zenia' under Section 18 of RERA. The complainant sought compensation for the delayed possession, non-availability of the occupancy certificate, and the failure to provide the agreed amenities as per the agreement for sale, including a school, hospital, and lift. The MahaRERA noted that the respondent failed to appear in hearings and decided the case ex-parte.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant sought compensation and interest under Section 18 of RERA, citing the delayed possession of the flat, non-provision of amenities as per the agreement, and failure of the respondent to obtain an occupancy certificate. Despite taking possession of the flat, the complainant is still paying EMI and has not occupied the flat."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondent did not appear in the hearings or provide any defense. The respondent failed to file a reply or respond to the complaint despite multiple hearings and notices."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA noted that the respondent failed to provide the agreed amenities, did not obtain the occupancy certificate, and did not attend hearings. The complainant’s request for compensation was reviewed, and the case was transferred to the Adjudicating Officer for further action under Section 18(3) of RERA."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA decided the case ex-parte due to the respondent's failure to attend hearings or submit a defense. The case was transferred to the Adjudicating Officer for determining the quantum of compensation under Section 18(3) of RERA."
          }
        ],
        "strong_points": [
          "The complainant emphasized the failure of the respondent to provide essential amenities and the lack of an occupancy certificate, which are significant issues under RERA.",
          "MahaRERA's decision to proceed ex-parte highlighted the respondent's lack of participation in the proceedings."
        ],
        "weak_points": [
          "The complainant had already taken possession of the flat in 2015, which could have weakened the case for compensation based on delayed possession.",
          "The respondent did not provide a defense or appear at the hearings, which could imply non-compliance with MahaRERA’s procedural requirements."
        ]
      },
      {
        "case_id": "case_021",
        "case_title": "Complaint No. CC005000000053914 - Suchet P. Padhye vs Marvel Omega Builders Pvt. Ltd. & Marvel Realtors & Developers Pvt. Ltd.",
        "case_description": "Suchet P. Padhye and Zareen S. Padhye filed a complaint under Section 18 of RERA seeking interest and compensation due to a delay in possession of flat No. B-1102 in the project 'Marvel Cerise Building B,C & D', Pune. The complainants made full payment and were to receive possession by December 31, 2013, but possession was only delivered on March 1, 2019, after a delay of 1885 days. The respondents cited a supplementary agreement extending the possession date to September 2017, arguing for the dismissal of the complaint. MahaRERA transferred the case to the Adjudicating Officer for determination of the compensation.",
        "sections": [
            {
                "section_id": "section_001",
                "section_title": "Complainant’s Claims",
                "section_description": "The complainants sought compensation for the delay in possession of flat B-1102 in the project 'Marvel Cerise Building B,C & D'. They paid the full consideration of Rs. 1,28,97,500/- and were entitled to possession by December 31, 2013. However, the possession was delivered on March 1, 2019, after a delay of 1885 days. They sought compensation under Section 18 of RERA."
            },
            {
                "section_id": "section_002",
                "section_title": "Respondents’ Defense",
                "section_description": "The respondents argued that the parties had entered into a supplementary agreement for sale, which extended the possession date to September 2017 or until the completion certificate was obtained. They requested the dismissal of the complaint based on the supplementary agreement."
            },
            {
                "section_id": "section_003",
                "section_title": "Authority Observations",
                "section_description": "MahaRERA observed the delay in possession of the flat beyond the agreed date and noted that the complainants were entitled to seek compensation under Section 18(3) of RERA for the delay. Despite the respondents' defense regarding the supplementary agreement, MahaRERA determined that the case required further adjudication regarding the quantum of compensation."
            },
            {
                "section_id": "section_004",
                "section_title": "Decision",
                "section_description": "MahaRERA transferred the case to the Adjudicating Officer in Pune for further proceedings, specifically for the determination of compensation under Section 18 of RERA, as the complainants had experienced a significant delay in possession."
            }
        ],
        "strong_points": [
            "The complainants emphasized the significant delay of 1885 days in receiving possession, which is a major lapse on the part of the respondents.",
            "MahaRERA acknowledged the applicability of Section 18 of RERA, entitling the complainants to seek compensation."
        ],
        "weak_points": [
            "The respondents presented a supplementary agreement that extended the possession date, which could potentially weaken the complainants' case for compensation.",
            "The complainants had already taken possession of the flat, which may affect the outcome regarding the quantum of compensation."
        ]
    },
    {
        "case_id": "case_022",
        "case_title": "Complaint No. CC005000000022943 - Pravin S. Nar-lkarri vs Clover Developers Pvt. Ltd. & Anr.",
        "case_description": "Pravin S. Nar-lkarri filed a complaint challenging the construction of buildings Nos. 8 & 9 on land meant for common amenities in the 'Clover Highlands' project. The complainant, an allottee in Building No. 21, contended that his undivided share in the common land was violated, and that the consumption of full FSI for construction was illegal. He sought redress under Section 31 of RERA, citing violations of Maharashtra Real Estate (Regulation and Development) Rules, 2017. However, the case was complicated by a pending civil suit on the same issue and the respondent’s contention that the matter was sub judice.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant claimed that as an allottee of Building No. 21 in the 'Clover Highlands' project, he held undivided interest in the land meant for common amenities. He objected to the construction of Buildings Nos. 8 & 9 on this land, asserting that the respondent had consumed the entire FSI, violating his rights. He also cited contraventions of specific provisions of Maharashtra Real Estate (Regulation and Development) Rules, 2017."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondents contended that the complainant was not an allottee of the land in question (Buildings Nos. 8 & 9) and that the complaint was not maintainable. They claimed that the land for common amenities was under the control of the society, and the dispute regarding the construction of the buildings was part of a pending civil suit. They argued that the case should not be addressed under RERA."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the complainant's issue related to common amenities and FSI usage was also being litigated in civil court. While the complainant had an interest in the common land, the court had already dismissed related consumer complaints and writ petitions. MahaRERA indicated that the matter was best addressed in civil court under the Maharashtra Ownership Flats Act, 1963 (MOFA), rather than under RERA."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA dismissed the complaint, stating that the dispute over the construction of Buildings Nos. 8 & 9 and the use of common amenities land fell within the jurisdiction of the civil court. However, the case was maintained only concerning potential contraventions of RERA rules, specifically regarding the registered project."
          }
        ],
        "strong_points": [
          "The complainant had a legitimate concern regarding his undivided interest in the common land, raising important issues under Section 31 of RERA.",
          "The civil court’s involvement and the dismissal of related writ petitions did not preclude RERA’s examination of potential violations of rules related to the registered project."
        ],
        "weak_points": [
          "The complainant's case was complicated by the pending civil suit, which dealt with the same issues, reducing the clarity and effectiveness of the RERA complaint.",
          "MahaRERA found that the dispute over the common land and FSI usage was primarily within the jurisdiction of civil courts, weakening the case's standing under RERA."
        ]
      },
      {
        "case_id": "case_023",
        "case_title": "Complaint No. CC006000000195650 - Manju Somani & Gaurav Somani vs Raja Builders",
        "case_description": "Manju Somani and Gaurav Somani filed a complaint seeking a refund of the balance amount of Rs. 25,00,000, following the cancellation of a sale transaction for a flat in the 'Raja Builders' project. The complainants claim to have received a partial refund, but they are entitled to the remaining amount as per mutual understanding. The respondent argues that the entire consideration has been refunded and that the complaint is improperly filed with the MahaRERA Authority instead of the Adjudicating Officer.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants’ Claims",
            "section_description": "The complainants sought a refund of the remaining balance of Rs. 25,00,000 for the flat transaction, claiming the refund was incomplete despite an earlier partial refund. They argued that the respondent has violated the terms of mutual understanding regarding the sale cancellation."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondent claimed that the entire consideration amount had been refunded to the complainants, citing documentary evidence of the payments. They also argued that the complaint was incorrectly filed before the MahaRERA Authority instead of the Adjudicating Officer, as the matter pertained to a monetary claim."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA noted that while the respondent claims the entire amount has been refunded, this will be considered during the final hearing. The complaint was not rejected at this stage, and the authority maintained that the case should be evaluated further to determine the legitimacy of the refund claim."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA rejected the respondent's application to dismiss the complaint and allowed the case to proceed. The points raised by the respondent will be considered in the final hearing, but the complaint was not dismissed at this stage."
          }
        ],
        "strong_points": [
          "The complainants provided evidence of a partial refund and a claim for the remaining balance of Rs. 25,00,000.",
          "MahaRERA recognized the need for a full evaluation of the respondent's defense at the final hearing."
        ],
        "weak_points": [
          "The respondent's defense, supported by documentary evidence, indicated that the full amount had already been refunded.",
          "The complaint was filed with the MahaRERA Authority rather than the Adjudicating Officer, as required for monetary claims."
        ]
      },
      {
        "case_id": "case_024",
        "case_title": "Complaint No. CC006000000012041 - Sanjeev Sharma vs Ravi Developments",
        "case_description": "Sanjeev Sharma filed a complaint against Ravi Developments for failure to deliver possession of his apartment in the 'Gaurav Aster' project within the stipulated time frame. Although the construction up to the 10th floor was completed, the complainant has not received possession due to incomplete amenities and the lack of an occupancy certificate. The complainant seeks to have possession handed over and to claim interest for the delay.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant claims that despite completing construction up to the 10th floor, the Respondent has failed to hand over possession of the apartment, including the occupancy certificate. The complainant seeks possession of the apartment and interest on the delayed handover."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The Respondent explained that the construction work for the relevant floors was completed in 2013-2014, and all other allottees of these floors have taken possession. The Respondent stated that the remaining construction is underway and that possession has been repeatedly offered to the complainant."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA acknowledged that the construction for the complainant’s apartment was completed but noted that the lack of amenities and an occupancy certificate had delayed possession. The Authority emphasized that the Respondent must complete the project and obtain the occupancy certificate as per the registered timeline."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA directed the Respondent to complete the remaining construction work and obtain the occupancy certificate. The complainant was given the liberty to demand interest for the delay at an appropriate stage in accordance with Section 18 of RERA."
          }
        ],
        "strong_points": [
          "The Respondent acknowledged the completion of construction but failed to deliver the occupancy certificate, affecting possession.",
          "The complainant has the legal right under Section 18 to demand interest for the delay in possession."
        ],
        "weak_points": [
          "The Respondent claimed that the possession was offered multiple times but was not accepted by the complainant.",
          "The project was still under development, with amenities and certain floors unfinished, which contributed to the delay."
        ]
      },
      {
        "case_id": "case_025",
        "case_title": "Complaint No. CC006000000193924 - Jaikumar Gupta HUF vs Realgem Buildtech Pvt. Ltd.",
        "case_description": "Jaikumar Gupta HUF filed a complaint seeking a refund and compensation for the delay in possession of a flat in the project 'Rustomjee Crown - Phase I.' The complainant requested a refund under Section 18 of RERA, citing the delay in possession, along with interest and compensation. The MahaRERA noted that the complainant had filed two separate complaints for the same cause of action, leading to confusion about the relief sought. Ultimately, one complaint was withdrawn, and the other was retained for final hearing.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant sought a refund of the amounts paid for the flat under Section 18 of RERA, including taxes, stamp duties, and other charges. Additionally, the complainant requested compensation for the delay in possession, calculated as interest at 21% per annum. The complainant also sought damages for the inconvenience and mental agony caused by the delay, as well as legal and logistical costs."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondent raised the issue of maintainability, arguing that the agreement for sale included an arbitration clause, which should have been exercised first. Additionally, the respondent pointed out that the complainant had filed two complaints for the same premises and cause of action, one seeking possession with interest and the other seeking a refund."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed the complainant's filing of two separate complaints for the same cause of action and directed the complainant to explain the reason for this. The authority also took note of the arbitration clause in the agreement for sale and questioned the maintainability of the complaint. The complainant was absent at the hearing but later sought to withdraw one of the complaints."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA disposed of the second complaint (CC006000000195172) as withdrawn and directed that the first complaint (CC006000000193924) be listed for final hearing. No order was passed regarding costs, and the case was scheduled to be heard as per seniority."
          }
        ],
        "strong_points": [
          "The complainant raised valid concerns regarding the delay in possession and sought appropriate relief under Section 18 of RERA.",
          "MahaRERA provided clarity by allowing the complainant to retain one complaint for final hearing after addressing the confusion caused by two complaints."
        ],
        "weak_points": [
          "The respondent's argument about the arbitration clause in the agreement raised doubts about the appropriateness of the RERA forum for resolving the dispute.",
          "The complainant’s filing of two complaints for the same issue complicated the proceedings and led to procedural delays."
        ]
      },
      {
        "case_id": "case_026",
        "case_title": "Complaint No. CC006000000196910 - Krishnan Pich Mani & Sindhu Krishnan vs Nirmal Lifestyle (Kalyan) Private Limited",
        "case_description": "Krishnan Pich Mani and Sindhu Krishnan filed a complaint seeking directions from MahaRERA to refund the amounts paid for flat no. 704 in the 'Lifestyle City Kalyan - Platano E' project, citing a delay in possession. The respondent, Nirmal Lifestyle (Kalyan) Pvt. Ltd., was undergoing a Corporate Insolvency Resolution Process (CIRP), and the National Company Law Tribunal (NCLT) had imposed a moratorium on legal proceedings against the respondent. MahaRERA deferred its decision, pending the outcome of the insolvency proceedings, and adjourned the complaint sine die with liberty for the complainants to approach MahaRERA post-disposal of the insolvency case.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants' Claims",
            "section_description": "The complainants sought a refund of the amounts paid for the flat in the project 'Lifestyle City Kalyan - Platano E,' along with interest and compensation under RERA. They argued that despite the moratorium imposed by the NCLT, MahaRERA should still proceed with the complaint against the promoters for refund."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Defense",
            "section_description": "The respondent informed MahaRERA that the company was undergoing Corporate Insolvency Resolution Process (CIRP) as per an NCLT order, and a moratorium had been imposed under Section 14 of the Insolvency and Bankruptcy Code, preventing legal proceedings against it. The respondent requested that the complaint be kept in abeyance until the final disposal of the CIRP."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA noted the imposition of the moratorium by the NCLT, preventing the continuation of legal proceedings against the respondent. The authority also considered the Supreme Court judgment cited by the complainants, which stated that the moratorium only applies to the 'Corporate Debtor' and not its directors. However, since the complainants had not named the directors as respondents, MahaRERA could not proceed with the complaint at that stage."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA decided to adjourn the complaint sine die due to the moratorium imposed by the NCLT. The complainants were granted liberty to approach MahaRERA after the final disposal of the insolvency proceedings against the respondent."
          }
        ],
        "strong_points": [
          "The complainants highlighted the legal precedent from the Supreme Court, arguing that the moratorium does not prevent proceedings against the promoters directly.",
          "MahaRERA provided a fair hearing and acknowledged the legal constraints imposed by the NCLT proceedings."
        ],
        "weak_points": [
          "The complainants failed to include the directors of the respondent company in the complaint, limiting the scope of legal action available under RERA.",
          "Despite the complainants' arguments, MahaRERA could not proceed with the case due to the ongoing insolvency proceedings and the moratorium."
        ]
      },
      {
        "case_id": "case_027",
        "case_title": "Complaint No. CC006000000090122 - Jesa G. Patel (HUF) vs Swastik Realty Pvt. Ltd., RajendraLaxmanWani, KalpanaRajendraWani",
        "case_description": "Jesa G. Patel (HUF) filed a complaint seeking directions from MahaRERA to the respondents, Swastik Realty Pvt. Ltd., Rajendra Laxman Wani, and Kalpana Rajendra Wani, to execute an agreement for sale and pay interest for delayed possession under RERA. The complainant also sought compensation for mental agony due to delayed possession and false information provided by the respondents, including the sale of a flat to a third party without the complainant’s consent. The complaint was initially heard by MahaRERA, which transferred the case to the Ld. Adjudicating Officer for further adjudication and compensation under RERA.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Claims",
            "section_description": "The complainant sought directions from MahaRERA for the execution of an agreement for sale under Section 13 and requested compensation for delayed possession under Section 18 of the RERA. The complainant also sought compensation for mental agony under Section 12 due to false information and amendments to plans without consent, as well as the sale of a flat to a third party without the complainant’s consent."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Defense",
            "section_description": "The text does not provide specific details about the respondent's defense, but it is implied that the respondent may have disputed the claims of compensation and the issues raised by the complainant."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority's Observations",
            "section_description": "MahaRERA observed that the complainant's claim for compensation under Sections 12 and 14 of RERA fell under the jurisdiction of the Ld. Adjudicating Officer, who is authorized to adjudge compensation in such matters. The complainant had also stated that a flat had been sold to a third party without their consent."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the case to the Ld. Adjudicating Officer for further adjudication and decision on the compensation claim as per the provisions of RERA. The Adjudicating Officer was requested to decide the issues raised by the complainant at the earliest."
          }
        ],
        "strong_points": [
          "The complainant clearly stated the violations under Sections 12 and 14, highlighting the false information provided and unauthorized sale of the flat to a third party.",
          "MahaRERA promptly transferred the case to the appropriate authority (the Ld. Adjudicating Officer) for further action, ensuring that the complainant’s issues would be addressed."
        ],
        "weak_points": [
          "The text does not provide the respondent's defense, leaving the case somewhat one-sided in the information provided.",
          "The complaint was transferred without a final decision, which may cause a delay in resolution for the complainant."
        ]
      },
      {
        "case_id": "case_028",
        "case_title": "Complaint No. CC006000000396998 - MDP Partners (Mr. Paranjape) vs Avarsekar Realty Pvt Ltd",
        "case_description": "MDP Partners (Mr. Paranjape) filed a complaint before MahaRERA seeking directions to the respondent, Avarsekar Realty Pvt Ltd, to upload project-related documents as required under Sections 4 and 11 of the Real Estate (Regulation & Development) Act, 2016 (RERA) for the registered project 'Shristi Sea View' in Mumbai. During the proceedings, MahaRERA discovered that the respondent was undergoing insolvency proceedings under the Insolvency and Bankruptcy Code (IBC), 2016. As per the order from the National Company Law Tribunal (NCLT), a moratorium was imposed, preventing any legal proceedings against the respondent. Consequently, MahaRERA adjourned the case sine die, granting the complainant the liberty to approach MahaRERA after the insolvency case is resolved.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Claims",
            "section_description": "The complainant sought directions for the respondent to upload project-related documents as per Sections 4 and 11 of RERA concerning the 'Shristi Sea View' project. The complainant requested this action to ensure the respondent complied with RERA requirements regarding project documentation."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Defense",
            "section_description": "The text does not provide specific details about the respondent's defense. The respondent did not appear for the hearing, and the case was impacted by insolvency proceedings against the respondent."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority's Observations",
            "section_description": "MahaRERA discovered that the respondent was undergoing insolvency proceedings under the IBC, with a moratorium on legal proceedings against the respondent, as per the NCLT's order. This moratorium prohibited the continuation of legal proceedings under RERA."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA adjourned the complaint sine die in light of the NCLT's order and the moratorium. The complainant was granted the liberty to approach MahaRERA again after the final disposal of the insolvency proceeding against the respondent."
          }
        ],
        "strong_points": [
          "MahaRERA promptly identified the impact of the ongoing insolvency proceedings and took the appropriate legal action by deferring the case.",
          "The complainant was informed of the ongoing insolvency process, and the option to revisit the case after resolution was clearly communicated."
        ],
        "weak_points": [
          "The respondent's failure to appear for the hearing and the absence of any defense or explanation weakens the transparency of the case.",
          "The case cannot proceed until the insolvency proceedings are resolved, delaying a resolution for the complainant."
        ]
      },
      {
        "case_id": "case_029",
        "case_title": "Complaint No. CC006000000110754 - Rupinder Elavia vs Aryamaan Developers Private Limited",
        "case_description": "Rupinder Elavia filed a complaint before MahaRERA seeking a refund of the amount paid along with interest and compensation under Section 18 of the Real Estate (Regulation & Development) Act, 2016 (RERA) for the booking of a flat in Aryamaan Developers' project 'Centrona Nova B' in Ghatkopar East, Mumbai. The complainant alleged delays in possession and misleading information regarding the project completion date. Despite the interim order passed by MahaRERA on 08-10-2020 transferring the case to the Adjudicating Officer, the respondent claimed no obligation was violated. The matter was further complicated by an ongoing appeal, which resulted in the adjournment of the case sine die until the appeal was decided.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant's Claims",
            "section_description": "The complainant sought a refund of the amount paid, Rs. 22,26,137/-, along with interest and compensation due to delays in possession of the flat and misleading information provided by the respondent about the project completion date."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Defense",
            "section_description": "The respondent contended that the project completion date was set for 31/12/2022, and the complaint was premature since possession had not been due yet. They further argued that the complainant's email was not legally binding and denied any violation of RERA provisions. The respondent also pointed to an ongoing appeal challenging the interim order."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority's Observations",
            "section_description": "MahaRERA noted that an interim order had been passed on 08-10-2020, transferring the complaint to the Adjudicating Officer. The MahaRERA found that the complainant had challenged this order in an ongoing appeal before the Appellate Tribunal, making it impossible to proceed with the case until the appeal was decided."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA adjourned the complaint sine die until the final decision of the pending appeal. The parties were directed to inform MahaRERA once the appeal was resolved."
          }
        ],
        "strong_points": [
          "MahaRERA provided clear guidance on the procedural status of the case and the impact of the pending appeal on further proceedings.",
          "The complainant was granted a reasonable opportunity to pursue the appeal, keeping the case under review."
        ],
        "weak_points": [
          "The case is delayed due to the ongoing appeal, leaving the complainant without a final resolution for a prolonged period.",
          "The respondent's defense of no violation is unchallenged until the appeal is resolved, which may prolong the legal process."
        ]
      },
      {
        "case_id": "case_030",
        "case_title": "Complaint No. CC006000000056050 - Dhanraj Rajpal vs Benchmark Projects Pvt Ltd & Complaint No. CC006000000056049 - Neeta Rajpal vs Benchmark Projects Pvt Ltd",
        "case_description": "Dhanraj Rajpal and Neeta Rajpal filed separate complaints before MahaRERA, alleging that the respondent, Benchmark Projects Pvt Ltd, failed to hand over possession of commercial units in the project 'Benchmark Plaza' situated in Andheri, Mumbai, within the stipulated timeline. The complainants also alleged deficiencies in the carpet area of the commercial units and violations of BMC norms. They requested that the respondent rectify the violations, obtain the occupancy certificate (OC), and hand over possession. Additionally, they sought compensation for the delay in possession. The respondent argued that the delay was due to factors beyond their control and claimed that the complainants were already in possession and conducting business from the units. The case is ongoing, with the prayer for interest on delayed possession to be adjudicated once the project obtains the OC.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants' Claims",
            "section_description": "Dhanraj Rajpal and Neeta Rajpal claimed that the respondent failed to hand over possession of commercial units within the stipulated time and violated BMC norms. They also alleged deficiencies in the carpet area of the units. The complainants requested that the respondent rectify the BMC violations, obtain the OC, and hand over possession, along with compensation for the delayed possession."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent's Defense",
            "section_description": "The respondent argued that the project completion was delayed due to circumstances beyond their control. They also denied any deficiency in the carpet area of the units and stated that the complainants were already in possession of the units and conducting business. The respondent intended to obtain the part occupancy certificate for the floors covering the complainants' units."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority's Observations",
            "section_description": "MahaRERA noted that the respondent needed to complete the remaining work and obtain the OC according to the timeline stipulated on the project registration webpage. The authority indicated that the issue of interest for delayed possession would be adjudicated once the project obtained the OC."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA directed the respondent to complete the remaining project work and obtain the OC according to the timeline on the registration webpage. The request for interest on delayed possession would be adjudicated once the OC was obtained. The case was adjourned for a future hearing."
          }
        ],
        "strong_points": [
          "MahaRERA clarified that the issue of interest for delayed possession would be addressed at a later stage, ensuring the parties understood the procedural steps ahead.",
          "The authority took a balanced approach, allowing the respondent time to complete the project work and obtain the OC while addressing the complainants' concerns."
        ],
        "weak_points": [
          "The case has been delayed, and the complainants' request for interest on delayed possession is still pending, which could result in frustration for the parties involved.",
          "While the respondent intends to obtain the part occupancy certificate, the timeline for completion remains uncertain, prolonging the resolution of the case."
        ]
      },
      {
        "case_id": "case_031",
        "case_title": "Complaint No. CC005000000023401 - Pallavi Ambaldhage & Mangala Ambaldhage vs Amit Ashok Thepade, Deepak Amrutlal Gugale, Roshan Ramesh Dusane",
        "case_description": "Pallavi Ambaldhage and Mangala Ambaldhage filed a complaint seeking possession of flat no. 403 in the 'Galaxy One' project under Section 18 of RERA, citing non-delivery of possession by the promised date and deficiencies in construction. The respondents claimed that they had received the part occupancy certificate and offered possession, but the complainants refused to take possession without making the balance payment. The case was transferred to the Adjudicating Officer for compensation determination.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainants sought possession of the flat, along with interest for delayed possession and compensation under Section 18 of RERA. They argued that the respondents failed to deliver possession by the agreed date of 31/12/2015 and did not complete promised amenities, leading to breaches of RERA."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondents, through their advocate, stated that they had obtained the part occupancy certificate on 06/07/2018 and had offered possession to the complainants. However, the complainants had not made the balance payment and were unwilling to take possession of the flat."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA noted that the respondents had obtained the part occupancy certificate, but the complainants had raised concerns about defects in the construction. MahaRERA acknowledged the need for compensation due to the construction discrepancies and decided to transfer the case to the Adjudicating Officer in Pune for compensation determination."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA decided to transfer the complaint to the Ld. Adjudicating Officer in Pune to determine the issue of compensation under the provisions of RERA and its rules."
          }
        ],
        "strong_points": [
          "The complainants highlighted the delayed possession and construction defects, which violated the agreement and RERA provisions.",
          "The respondents' failure to fully comply with the promised amenities and construction quality warranted a compensation review."
        ],
        "weak_points": [
          "The respondents had received the part occupancy certificate, which could mitigate some of the complainants' claims for non-possession.",
          "The complainants refused to take possession despite the respondents offering it, citing unresolved payment issues."
        ]
      },
      {
        "case_id": "case_032",
        "case_title": "Non-Execution Application in Complaint No. CC005000000011612 - Mr. Tukaram Dhage vs M/s. Vastushilp Builders and Developers",
        "case_description": "Mr. Tukaram Dhage filed a non-execution application seeking a refund of the entire amount paid, along with interest and compensation under Section 18 of RERA. This was due to the respondent's failure to comply with the order dated 5th October 2018, which directed them to complete the pending work within three months. MahaRERA observed that the respondent neglected to fulfill the directions and transferred the case to the Adjudicating Officer for further decision on compensation.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainant sought a refund of the total amount paid, along with interest and compensation, citing the respondent’s failure to comply with the MahaRERA order dated 5th October 2018, which directed them to complete the pending work within three months."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondents’ Defense",
            "section_description": "The respondent did not appear during the hearing of the non-execution application and failed to provide any defense or explanation for their non-compliance with the MahaRERA order."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the respondent had failed to comply with the directions issued on 5th October 2018. Considering the circumstances, MahaRERA deemed the complainant’s request for a refund, interest, and compensation justified under Section 18 of RERA."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA directed that the case be transferred to the Ld. Adjudicating Officer in Pune for determining the quantum of compensation under Section 18 of RERA. The non-execution application was disposed of with this direction."
          }
        ],
        "strong_points": [
          "The complainant highlighted the respondent’s complete failure to comply with a binding MahaRERA order, justifying their claim for a refund and compensation.",
          "MahaRERA acknowledged the non-compliance and transferred the case for appropriate adjudication under Section 18 of RERA."
        ],
        "weak_points": [
          "The respondent did not appear to present any defense, leaving the complaint unchallenged during the hearing.",
          "There is no evidence in the provided text indicating that the respondent attempted partial compliance or sought an extension."
        ]
      },
      {
        "case_id": "case_033",
        "case_title": "Complaint Nos. CC006000000192257 & CC006000000251688 - Megha Prashant Jadhav and Prasad Prasannan Poozhipurathu vs M/s. Enkay Castle",
        "case_description": "Megha Prashant Jadhav and Prasad Prasannan Poozhipurathu filed two separate complaints seeking a refund of the full amount paid for their respective flats in the project 'Enkay Garden – Bluebell,' along with interest under Section 18 of RERA. The complainants argued that the project is at a standstill, and the registration with MahaRERA has not been extended. Although the respondent failed to appear for the hearing, MahaRERA granted them one final opportunity to present their case, adhering to the principles of natural justice.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants’ Claims",
            "section_description": "The complainants sought a refund of the total amount paid for their respective flats, along with interest, citing that the project is at a standstill and the MahaRERA registration has not been extended. They emphasized their right to withdraw from the project due to non-compliance by the respondent."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent’s Defense",
            "section_description": "The respondent failed to appear during the initial hearing on 06-09-2022 but later submitted a written application claiming that they had not received the hearing notice or virtual hearing link. They requested another opportunity to present their defense."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA verified that the hearing notice and virtual hearing link were duly served to the respondent on 24-08-2022 and 05-09-2022, respectively, disproving the respondent's claim of non-receipt. However, in adherence to the principles of natural justice, MahaRERA granted the respondent one final opportunity to submit their reply and participate in the next hearing."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA modified the Roznama dated 06-09-2022 and scheduled a final hearing for 12-10-2022. The respondent was directed to upload their reply and written submissions within one week of the interim order. The case was deferred for final resolution after hearing both parties."
          }
        ],
        "strong_points": [
          "The complainants highlighted the project’s standstill and lack of registration extension, which are significant breaches of RERA provisions.",
          "MahaRERA ensured due process by granting the respondent one last opportunity to represent their case."
        ],
        "weak_points": [
          "The respondent’s absence during the initial hearing delayed the resolution of the complaints.",
          "The respondent's claim of not receiving the hearing notice, though disproven, could be seen as an attempt to delay the proceedings further."
        ]
      },
      {
        "case_id": "case_034",
        "case_title": "Complaint Nos. CC006000000054933 to CC006000000100300 - Gita Devi Modi & Others vs Kavya Mira Realty",
        "case_description": "Multiple complainants filed complaints against Kavya Mira Realty regarding delays in possession of apartments in the project 'Kavya Residency' located at Ghodbunder Road, Thane. The complainants alleged that the respondent failed to hand over possession within the stipulated period as per the agreements. They sought relief under Section 18 of the Real Estate (Regulation and Development) Act, 2016. Despite claims of construction completion, the respondent cited external factors, such as the lack of a ULC NOC, preventing them from obtaining the Occupancy Certificate.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainant’s Claims",
            "section_description": "The complainants alleged delays in possession of apartments purchased in the respondent’s project 'Kavya Residency.' They sought interest on the delay under Section 18 of RERA, as the possession date mentioned in the agreements had long passed. Some complainants also noted that alternative accommodation arrangements were made."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent’s Defense",
            "section_description": "The respondent’s advocate explained that the delay was caused by circumstances beyond their control, particularly the inability to obtain a ULC NOC from the Thane Collector. It was also noted that some complainants had taken possession for fit-out purposes, but the Occupancy Certificate could not be acquired due to these challenges."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the respondent had failed to adhere to prior orders to hand over possession by March 31, 2018, and pay interest to the complainants. Additionally, the respondent’s project registration had expired on December 31, 2019, and the statutory provisions of RERA were not complied with. The authority stressed the importance of protecting the rights of the allottees."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA directed the respondent to provide a list of allottees and their contact details to the complainants within 30 days to enable them to form an Association of Allottees (AOA). The AOA may proceed under Sections 7/8 of RERA for project completion. Furthermore, the respondent was ordered to upload the Form 5 Annual Audit Report for FY 18-19 within 30 days and maintain the status quo regarding alternate accommodation arrangements. The respondent was also restrained from creating third-party rights for unsold apartments in the project."
          }
        ],
        "strong_points": [
          "MahaRERA proactively directed the formation of an Association of Allottees to safeguard their rights and ensure informed decision-making for project completion.",
          "The authority emphasized the respondent’s non-compliance with statutory provisions, such as project registration and prior orders."
        ],
        "weak_points": [
          "The delays caused by the lack of a ULC NOC and the inability to obtain an Occupancy Certificate prolonged the complainants’ wait for relief.",
          "The complainants’ ability to secure immediate possession or compensation is hindered by the ongoing administrative and legal challenges."
        ]
      },
      {
        "case_id": "case_035",
        "case_title": "Complaint Nos. CC006000000192880, CC006000000192882 - Daksha Shailesh Karkrech, Shailesh Kakrecha, Jatin Chunilal Gangar vs M/s. Swadhinta Builders LLP",
        "case_description": "The complainants, Mrs. Daksha Shailesh Karkrech, Mr. Shailesh Kakrecha, and Mr. Jatin Chunilal Gangar, filed separate complaints seeking possession of their flats along with interest and compensation for delayed possession under section 18 of the Real Estate (Regulation & Development) Act, 2016. The project in question, 'Shri Vallabh Residency,' registered under MahaRERA No. P51800005762, has been abandoned by the respondent, who expressed willingness to refund the amounts paid along with interest. The complainants, however, seek possession of their flats.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants’ Claims",
            "section_description": "The complainants sought possession of their flats in the 'Shri Vallabh Residency' project, along with interest and compensation for the delay under section 18 of RERA. They argued that the agreed possession dates had lapsed, and the respondent failed to deliver the flats as per the agreements."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent’s Position",
            "section_description": "The respondent, M/s. Swadhinta Builders LLP, admitted to abandoning the project and expressed readiness to refund the full amounts paid by the complainants along with interest. However, the complainants refuted this offer, insisting on possession of their flats."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA acknowledged that the complaints pertain to delayed possession and compensation under section 18 of RERA. However, as per sections 71 and 72 of RERA, the adjudication of compensation and interest falls under the jurisdiction of the Adjudicating Officer."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the complaints to the Adjudicating Officer for adjudication of the quantum of compensation and interest. Both parties were directed to proceed as per the provisions of RERA and the rules and regulations made thereunder."
          }
        ],
        "strong_points": [
          "MahaRERA facilitated a structured process by transferring the complaints to the Adjudicating Officer for proper adjudication.",
          "The respondent demonstrated a willingness to refund the amounts paid by the complainants along with interest."
        ],
        "weak_points": [
          "The abandonment of the project left the complainants without possession of their flats, causing significant distress and uncertainty.",
          "The transfer of the complaints to the Adjudicating Officer delays resolution and the potential relief sought by the complainants."
        ]
      },
      {
        "case_id": "case_036",
        "case_title": "Complaint No. CC006000000192855 - Nirav Laxmichand Nandu and Sejal Nirav Nandu vs Mr. Dhiren V Chheda",
        "case_description": "Mr. Nirav Laxmichand Nandu and Mrs. Sejal Nirav Nandu filed a complaint seeking a refund of the full amount paid for flat no. 6001 in the respondent’s project 'Horizon Homes' at Malad West, along with interest and compensation under section 18 of the Real Estate (Regulation & Development) Act, 2016. The respondent failed to deliver possession of the flat by the agreed date of 04-11-2017 and has not provided justified reasons for the delay. The project remains incomplete, and no occupancy certificate has been obtained.",
        "sections": [
          {
            "section_id": "section_001",
            "section_title": "Complainants’ Claims",
            "section_description": "The complainants sought a refund of Rs. 91,00,000 paid for flat no. 6001 in the 'Horizon Homes' project, along with interest and compensation for the delayed possession. They highlighted that the respondent failed to hand over possession by the agreed date of 04-11-2017 as per the registered agreement for sale dated 04-11-2015."
          },
          {
            "section_id": "section_002",
            "section_title": "Respondent’s Position",
            "section_description": "The respondent did not contest the claim for adjudication of compensation but failed to provide any valid justification for the delay in possession. The project remains incomplete, and no occupancy certificate has been obtained."
          },
          {
            "section_id": "section_003",
            "section_title": "Authority Observations",
            "section_description": "MahaRERA observed that the respondent was obligated to hand over possession of the flat by 04-11-2017, as per the agreement. However, the respondent failed to fulfill this obligation, and no valid reasons were provided for the delay. The authority acknowledged the substance in the complainants’ claim for a refund under section 18 of RERA."
          },
          {
            "section_id": "section_004",
            "section_title": "Decision",
            "section_description": "MahaRERA transferred the complaint to the Adjudicating Officer for adjudication of the quantum of compensation and interest as per the provisions of RERA and the rules and regulations made thereunder."
          }
        ],
        "strong_points": [
          "MahaRERA recognized the validity of the complainants’ claim for a refund under section 18 of RERA.",
          "The transfer of the complaint to the Adjudicating Officer ensures a focused adjudication of the quantum of compensation and interest."
        ],
        "weak_points": [
          "The respondent’s failure to provide valid reasons for the delay caused distress and uncertainty for the complainants.",
          "The project remains incomplete, and the lack of an occupancy certificate adds to the complainants’ concerns regarding the resolution timeline."
        ]
      },
      {
        "case_id": "case_037",
        "case_title": "Complaint No. CC005000000010924 - Ghevarchand Ganesmal Dhumavat & Meena Ghevarchand Dhumavat vs Spire Associates, Om Estate Developers Pvt. Ltd. & Mrs. Armaity Jimmy Sukhia",
        "case_description": "The complainants, home buyers of the 'Citywoods Pine' project, filed a complaint seeking a refund and interest due to delays in possession. They seek to amend their reliefs, requesting possession instead of a refund after an appellate order was passed remanding the matter for fresh consideration. The respondent contests the amendment, emphasizing that the complainants had previously sought a refund and terminated the agreement.",
        "sections": [
            {
                "section_id": "section_001",
                "section_title": "Complainant’s Claims",
                "section_description": "The complainants are seeking a refund of Rs. 1,61,10,705 along with interest of Rs. 23,30,068.59 due to the failure to hand over possession of their flat in the 'Citywoods Pine' project as per the terms of the agreement. They are now seeking possession and interest on delayed handover after the appellate order remanded the case."
            },
            {
                "section_id": "section_002",
                "section_title": "Respondent’s Defense",
                "section_description": "The respondents argue that the complainants had previously terminated the agreement and sought a refund, which was granted under the final order passed in 2018. They further state that the complainants' attempt to change their reliefs to possession is an attempt to gain financially beneficial relief by misusing legal provisions."
            },
            {
                "section_id": "section_003",
                "section_title": "Authority Observations",
                "section_description": "MahaRERA emphasized that the complainants had previously sought a refund and that the case has already traveled through multiple stages of litigation. The authority highlighted that Section 18 of the Real Estate (Regulation and Development) Act, 2016 provides complainants the choice between a refund or possession but questioned whether this choice could be exercised after such a prolonged litigation process. The request for amendment of reliefs was not allowed."
            },
            {
                "section_id": "section_004",
                "section_title": "Final Order",
                "section_description": "The MahaRERA Authority decided that the complaint will be adjudicated on the original reliefs (refund and interest). The request to amend the reliefs to possession was denied. The matter is scheduled for hearing on merits on 14.05.2024. The respondents are also directed to seek an extension for the project within 30 days."
            }
        ]
    },
    {
      "case_id": "case_038",
      "case_title": "Complaint No. CC006000000100497 - Mrs. Roopa N. Hegde and Mr. Narahari V. Hegde vs M/s. Sanvo Resort Pvt. Ltd",
      "case_description": "Mrs. Roopa N. Hegde and Mr. Narahari V. Hegde filed a complaint seeking possession of flat No. 2005, 20th floor, S-1 Wing-B, in the project 'Marathon Nexzone Zenith' along with interest and compensation for delayed possession under Section 18 of the Real Estate (Regulation and Development) Act, 2016 (RERA). The case was heard through video conferencing, and both parties presented their submissions. The complainants requested the case be transferred to the Ld. Adjudicating Officer/MahaRERA for adjudging compensation. The respondent had no objection to this transfer. The case has been transferred for further proceedings as per the provisions of RERA.",
      "sections": [
        {
          "section_id": "section_001",
          "section_title": "Complainant’s Claims",
          "section_description": "The complainants seek the possession of their flat No. 2005, 20th floor, S-1 Wing-B in the project 'Marathon Nexzone Zenith' along with interest and compensation under Section 18 of RERA due to the delayed possession. The complainants have requested that the case be transferred to the Ld. Adjudicating Officer for adjudication of compensation."
        },
        {
          "section_id": "section_002",
          "section_title": "Respondent’s Response",
          "section_description": "The respondent, M/s. Sanvo Resort Pvt. Ltd., has given no objection to transferring the case to the Ld. Adjudicating Officer of MahaRERA for a decision regarding the compensation under Section 18 of RERA."
        },
        {
          "section_id": "section_003",
          "section_title": "Authority Observations",
          "section_description": "MahaRERA has acknowledged the complaint and noted that the complainants have requested a transfer of the case to the Ld. Adjudicating Officer for compensation adjudication due to delayed possession under Section 18 of RERA."
        },
        {
          "section_id": "section_004",
          "section_title": "Decision",
          "section_description": "In light of the complainants' request and the respondent's agreement, the case has been transferred to the Ld. Adjudicating Officer/MahaRERA, Mumbai for adjudication as per RERA provisions."
        }
      ],
      "strong_points": [
        "The complainants are pursuing appropriate legal action by requesting the case be transferred for a decision on compensation under RERA.",
        "Both parties have agreed on the transfer of the case to the Ld. Adjudicating Officer for further proceedings."
      ],
      "weak_points": [
        "The case has yet to be decided regarding the compensation and possession, as it is being transferred to another authority for adjudication.",
        "The delayed possession of the flat continues to be a point of contention for the complainants, pending the adjudication decision."
      ]
    },
    {
      "case_id": "case_039",
      "case_title": "Complaint No. CC006000000012721 - Michael Raj vs Sultan Akbar Maldar",
      "case_description": "Michael Raj filed a complaint seeking possession of his flat No. B4-404 in the project 'Yashdeep Residency,' along with interest for delayed possession under Section 18 of RERA. The matter was initially referred to the Conciliation Forum, where both parties agreed to an amicable settlement. The case was subsequently transferred back to MahaRERA after settlement. However, the consent terms were not filed within the stipulated time, leading to the non-execution application being heard. The case has been adjourned for further hearing on 7th February 2024.",
      "sections": [
        {
          "section_id": "section_001",
          "section_title": "Complainant’s Claims",
          "section_description": "The complainant sought possession of his flat No. B4-404 in 'Yashdeep Residency' along with interest under Section 18 of RERA due to delayed possession. He also sought an order from MahaRERA directing the respondent to comply with the terms of settlement."
        },
        {
          "section_id": "section_002",
          "section_title": "Respondent’s Status",
          "section_description": "The respondent, Sultan Akbar Maldar, had agreed to the settlement of the matter before the Conciliation Forum. However, the respondent failed to file the consent terms within the one-month period granted by MahaRERA, leading to the non-execution application."
        },
        {
          "section_id": "section_003",
          "section_title": "Authority Observations",
          "section_description": "MahaRERA noted that while both parties had initially agreed to settle the matter amicably, the failure to file consent terms has led to the non-execution application. The MahaRERA found that the case must now proceed for regular hearing, with the next hearing scheduled for 7th February 2024."
        },
        {
          "section_id": "section_004",
          "section_title": "Decision",
          "section_description": "MahaRERA, upon reviewing the situation, has decided that the non-execution application has become infructuous. The original complaint is scheduled for hearing on 7th February 2024 for a final decision on the merits."
        }
      ],
      "strong_points": [
        "Both parties initially agreed to an amicable settlement before the Conciliation Forum.",
        "MahaRERA has scheduled a final hearing to resolve the matter, ensuring due process."
      ],
      "weak_points": [
        "The failure of the respondent to file the consent terms delayed the resolution of the case.",
        "The case is still pending, and the complainant's issue remains unresolved until the next hearing on 7th February 2024."
      ]
    },
    {
      "case_id": "case_040",
      "case_title": "Complaint No. CC006000000079254 - Gilbert Joseph Menezes vs Conwood Realty Private Limited",
      "case_description": "Gilbert Joseph Menezes and Suhas Surendra Pai filed complaints against Conwood Realty Private Limited, seeking possession of their apartments in the project 'Conwood Enclave' located at Thane. The complainants claimed that the respondent failed to hand over possession as per the agreements for sale. The respondent cited delays due to mitigating circumstances but expressed willingness to settle. MahaRERA deferred the decision, awaiting further project completion.",
      "sections": [
        {
          "section_id": "section_001",
          "section_title": "Complainant’s Claims",
          "section_description": "The complainants sought possession of their apartments in the 'Conwood Enclave' project and demanded interest for the delay, citing that the possession dates stipulated in their agreements for sale had long passed."
        },
        {
          "section_id": "section_002",
          "section_title": "Respondent’s Status",
          "section_description": "The respondent acknowledged the delay in project completion, attributing it to mitigating circumstances beyond their control. The respondent indicated that the project is 15 floors complete and plans to apply for the part occupation certificate soon, which includes the complainants' apartments."
        },
        {
          "section_id": "section_003",
          "section_title": "Authority Observations",
          "section_description": "MahaRERA noted the project’s current status and stated that it would be premature to decide on the delay and interest under Section 18 of the RERA Act until the project nears completion or is completed. The complainants and respondents were urged to continue settlement talks."
        },
        {
          "section_id": "section_004",
          "section_title": "Decision",
          "section_description": "MahaRERA deferred the decision, indicating that the delay and interest calculation will be addressed once the project is completed or close to completion. The next hearing will be scheduled accordingly."
        }
      ],
      "strong_points": [
        "The respondent has acknowledged the delay and is taking steps to complete the project.",
        "MahaRERA has provided a clear approach to dealing with project delays under Section 18 of RERA once the project is nearing completion."
      ],
      "weak_points": [
        "The complainants' possession remains delayed with no immediate resolution due to the ongoing project completion.",
        "The respondent’s claim of mitigating circumstances may extend the delay further, affecting the complainants’ ability to move in or seek compensation."
      ]
    },
    {
      "case_id": "case_041",
      "case_title": "Complaint No. CC005000000022153 - Vishal Ravindra Madan Bhakti Manohar Kawali vs Darode Jog Realities Pvt Ltd",
      "case_description": "Vishal Ravindra Madan Bhakti Manohar Kawali filed a complaint seeking a refund of the amount paid for flat no. A-303 in the project 'Shriniwas Liviano Phase-1,' along with interest and compensation for delayed possession under Section 18 of RERA. During the pendency of the complaint, the complainants took possession of the flat but continued to seek interest for the delayed possession of 56 months. The respondent had transferred the project to M/s. Goel Ganga, and MahaRERA transferred the case for compensation determination to the Adjudicating Officer.",
      "sections": [
        {
          "section_id": "section_001",
          "section_title": "Complainant’s Claims",
          "section_description": "The complainants sought a refund of the full amount paid for flat no. A-303 in 'Shriniwas Liviano Phase-1,' along with interest and compensation for the 56-month delay in possession, as per Section 18 of the RERA. They also highlighted that the respondent transferred the project to M/s. Goel Ganga during the pendency of the complaint."
        },
        {
          "section_id": "section_002",
          "section_title": "Respondent’s Status",
          "section_description": "The respondent did not file any reply with MahaRERA. The project had been transferred to M/s. Goel Ganga during the pendency of the case, complicating the resolution of the dispute."
        },
        {
          "section_id": "section_003",
          "section_title": "Authority Observations",
          "section_description": "MahaRERA observed that the respondent had failed to provide possession of the flat on the agreed date as per the agreement for sale. MahaRERA referred to Section 18(3) of the RERA, emphasizing that the complainants are entitled to compensation for delayed possession."
        },
        {
          "section_id": "section_004",
          "section_title": "Decision",
          "section_description": "MahaRERA transferred the case to the Adjudicating Officer for a decision regarding the compensation for the delayed possession. The Adjudicating Officer is tasked with determining the quantum of compensation as per the provisions of RERA."
        }
      ],
      "strong_points": [
        "The complainants have continued to seek relief even after taking possession, highlighting their concern for the delayed possession period of 56 months.",
        "MahaRERA referred the case to the Adjudicating Officer to ensure the appropriate determination of compensation under Section 18 of RERA."
      ],
      "weak_points": [
        "The respondent failed to file a reply, leaving the complainants’ claims unaddressed by the respondent.",
        "The transfer of the project to M/s. Goel Ganga during the pendency of the case could lead to delays in determining responsibility for the compensation."
      ]
    },
    {
      "case_id": "case_042",
      "case_title": "Complaint No. CC004000000010069 - Mr. Saket Sunil Gothe vs Mr. Anil Singh (Hagwood Commercial Developers Pvt Ltd)",
      "case_description": "Mr. Saket Sunil Gothe filed a complaint seeking a refund of the entire amount paid for flat no. 1104 in the project 'Prozone Palms,' along with interest due to delayed possession under Section 18 of the RERA. The complaint was transferred to the Ld. Adjudicating Officer, MahaRERA, and an initial ruling was passed in favor of the complainant. However, the respondent appealed to the Maharashtra Real Estate Appellate Tribunal, which remanded the matter back to MahaRERA for fresh adjudication.",
      "sections": [
        {
          "section_id": "section_001",
          "section_title": "Complainant’s Claims",
          "section_description": "The complainant sought a refund of the entire amount paid for flat no. 1104 in the 'Prozone Palms' project, along with interest, under Section 18 of RERA, citing delayed possession and violation of the terms in the agreement for sale."
        },
        {
          "section_id": "section_002",
          "section_title": "Respondent’s Defense",
          "section_description": "The respondent contested the claim, asserting that the original ruling for refund with interest under Section 18 was flawed. They argued that the claim for a refund of stamp duty was invalid and could not be amended at this stage, especially since the matter was remanded back by the Maharashtra Real Estate Appellate Tribunal."
        },
        {
          "section_id": "section_003",
          "section_title": "Authority’s Observations",
          "section_description": "MahaRERA reviewed the arguments and noted that the complainant could not seek a refund of the stamp duty under Section 18 of RERA. The authority observed that the provision does not allow for such a refund, though it could be considered for compensation under other sections of RERA."
        },
        {
          "section_id": "section_004",
          "section_title": "Decision",
          "section_description": "MahaRERA rejected the complainant's amendment request to include the refund of stamp duty, citing that Section 18 of RERA does not cover such claims. The case was adjourned for a final decision after hearing both parties on 21-09-2021."
        }
      ],
      "strong_points": [
        "The complainant sought a refund under a clear legal provision (Section 18 of RERA) for delayed possession, which is a strong basis for the claim.",
        "The matter was remanded by the Appellate Tribunal for further adjudication, providing the complainant with a chance for re-evaluation of the case."
      ],
      "weak_points": [
        "The amendment to include a refund of stamp duty was rejected, limiting the complainant’s claim for additional compensation.",
        "The matter's complexity has been extended due to the remand, and the final decision will depend on further proceedings, which may take time."
      ]
    },
    {
      "case_id": "case_043",
      "case_title": "Complaint Nos. CC006000000209739 and 3 Other Complaints - Sandeep R. Jain & Others vs Heritage Housing Corporation & Kersi Jamshed Randeria",
      "case_description": "Complainants filed complaints seeking directions from MahaRERA for the execution of agreements for sale, possession handover, and compensation for delayed possession in the 'Heritage Ahura Residency' project. However, issues concerning the project's commencement certificate and registration led to MahaRERA adjourning the complaints sine die until these matters are resolved.",
      "sections": [
          {
              "section_id": "section_001",
              "section_title": "Complainants' Claims",
              "section_description": "The complainants sought directions from MahaRERA to execute agreements for sale, handover possession, and compensate for the delayed possession of flats in the Heritage Ahura Residency project, under the provisions of sections 13 and 18 of RERA."
          },
          {
              "section_id": "section_002",
              "section_title": "Respondents' Status",
              "section_description": "The respondents failed to appear for hearings despite notice. The respondent no. 2, a promoter, had registered the project with MahaRERA, but the commencement certificate had expired and the project registration lapsed in December 2023 without obtaining an extension."
          },
          {
              "section_id": "section_003",
              "section_title": "Authority Observations",
              "section_description": "MahaRERA acknowledged the lapsing of the project registration and the expiry of the commencement certificate. The issue of project ownership and the lack of resolution regarding the dispute between the promoters of the project prevented further action on the complainants' requests."
          },
          {
              "section_id": "section_004",
              "section_title": "Decision",
              "section_description": "MahaRERA decided to adjourn the complaints sine die until the following issues are resolved: i) confirmation of the new promoter, ii) validation of the last commencement certificate, and iii) extension of the project registration. The complainants are permitted to approach MahaRERA once these issues are resolved."
          }
      ],
      "strong_points": [
          "The complainants have provided strong documentation for their claims under RERA for execution of agreements and compensation.",
          "MahaRERA has issued interim orders to protect the complainants' rights by restraining the respondents from creating third-party rights in respect of the complainants' allotments."
      ],
      "weak_points": [
          "The case is pending resolution due to issues regarding the commencement certificate and project registration, which may delay any substantive relief for the complainants.",
          "The failure of the respondents to appear in hearings and the dispute between the promoters could significantly prolong the case."
      ]
  },
  {
    "case_id": "case_044",
    "case_title": "Complaint No. CC006000000374839 - Gorachand Adhikari & Anr. vs Era Realtors Private Limited & Anr.",
    "case_description": "Gorachand Adhikari and two other complainants filed complaints seeking directions under RERA in respect of their flats in the project 'Alta Monte and Signet' located in Borivali, Mumbai. The complaints were clubbed together and a hearing was held, with both parties submitting written arguments. The respondent raised a preliminary issue regarding the maintainability of the complaints due to an arbitration clause in the agreements for sale. The MahaRERA has decided to await the decision of the Hon’ble Appellate Tribunal on a similar matter before proceeding further.",
    "sections": [
      {
        "section_id": "section_001",
        "section_title": "Complainants’ Claims",
        "section_description": "The complainants sought relief under the provisions of RERA concerning the booking of flats in the respondent’s project, 'Alta Monte and Signet,' citing issues under RERA."
      },
      {
        "section_id": "section_002",
        "section_title": "Respondent’s Argument",
        "section_description": "The respondent raised an issue regarding the maintainability of the complaints, citing a specific clause in the agreement for sale that requires disputes to be referred to arbitration."
      },
      {
        "section_id": "section_003",
        "section_title": "Authority Observations",
        "section_description": "MahaRERA observed that a similar issue regarding the maintainability of complaints due to an arbitration clause is under consideration by the Hon’ble Appellate Tribunal in an appeal involving the same project."
      },
      {
        "section_id": "section_004",
        "section_title": "Decision",
        "section_description": "MahaRERA decided to defer the decision on the maintainability of these complaints until the Hon’ble Appellate Tribunal resolves the pending appeal, which will have a bearing on the outcome of these complaints."
      }
    ],
    "strong_points": [
      "MahaRERA is awaiting a decision from the Hon’ble Appellate Tribunal, which may provide clarity on the maintainability of the complaints.",
      "The complaints are being heard collectively, which may expedite the process once the Appellate Tribunal's decision is made."
    ],
    "weak_points": [
      "The arbitration clause in the agreement may complicate the legal process and delay the resolution of the complaints.",
      "The decision on maintainability is currently pending, leaving the complainants in a state of uncertainty."
    ]
  },
  {
    "case_id": "case_045",
    "case_title": "Complaint No. CC006000000197691 - Vijay Mishrani vs Supreme Construction and Developers Pvt Ltd",
    "case_description": "Vijay Mishrani filed a complaint seeking directions from MahaRERA to the respondent to pay interest for delayed possession under Section 18 of the Real Estate (Regulation & Development) Act, 2016. The complainant purchased flat no. 2105 in the respondent’s project ‘Clan City- Aqua Marine’ but has faced multiple delays in possession, leading to financial burden and the demand for compensation under RERA provisions. The respondent cited external causes and force majeure situations as reasons for the delay, but MahaRERA has directed the matter to be heard with other related complaints.",
    "sections": [
      {
        "section_id": "section_001",
        "section_title": "Complainant’s Claims",
        "section_description": "The complainant sought directions from MahaRERA to direct the respondent to pay interest for delayed possession under Section 18 of RERA. The complainant expressed dissatisfaction with the delay in possession, the unilaterally changed possession date, and the respondent’s failure to adhere to timelines, which caused significant financial strain on the complainant."
      },
      {
        "section_id": "section_002",
        "section_title": "Respondent’s Defense",
        "section_description": "The respondent denied the complainant’s claims, stating that the construction delays were due to external causes such as administrative issues, force majeure, and delays in approvals from MMRDA and other government agencies. The respondent also argued that the complainant was aware of these issues and that the delay in possession was not under their control."
      },
      {
        "section_id": "section_003",
        "section_title": "Authority Observations",
        "section_description": "MahaRERA acknowledged the submission of both parties and noted that the delays were partly due to external factors beyond the respondent's control, such as issues with MMRDA and other authorities. MahaRERA also referred to related complaints filed by other allottees of the same project and highlighted the need for these reports to be reviewed before proceeding."
      },
      {
        "section_id": "section_004",
        "section_title": "Decision",
        "section_description": "MahaRERA directed that all complaints related to the respondent's ‘Clan City’ project be heard together, to ensure a comprehensive review of the claims from all affected parties. The matter was adjourned until further investigation into the reports from relevant authorities."
      }
    ],
    "strong_points": [
      "MahaRERA has acknowledged the external factors contributing to the delay, ensuring the complainant's concerns are addressed in the broader context of the project's issues.",
      "The case has been consolidated with other related complaints, ensuring a unified decision on the project’s delays."
    ],
    "weak_points": [
      "The delay in possession has caused significant financial strain on the complainant, with interest and penalties yet to be determined.",
      "The respondent’s reliance on external factors and force majeure may delay the resolution of the complaint, preventing immediate relief for the complainant."
    ]
  },
  {
    "case_id": "case_046",
    "case_title": "Complaint No. CC006000000057837 - Mrs. Harshada Mukund Pande vs M/s. Balaji Construction",
    "case_description": "Mrs. Harshada Mukund Pande and Mr. Mukund Madhukar Pande filed a complaint seeking directions from MahaRERA for interest and compensation under Section 18 of the Real Estate (Regulation & Development) Act, 2016, for delayed possession of flat no. 1003 in the respondent's project 'Trinity Oasis Phase II' in Thane. The complainants were seeking relief after repeated hearings, orders for refund, and claims for compensation due to delays in possession.",
    "sections": [
      {
        "section_id": "section_001",
        "section_title": "Complainants’ Claims",
        "section_description": "The complainants sought compensation and refund under Section 18 of the RERA for delayed possession of flat no. 1003 in 'Trinity Oasis Phase II,' and claimed the right to interest on amounts paid."
      },
      {
        "section_id": "section_002",
        "section_title": "Respondent’s Position",
        "section_description": "M/s. Balaji Construction, represented by Mr. Anil Kursija, appeared in person and contested the complaint. Despite repeated hearings, the complainants sought compensation and refund due to the delays caused by the respondent in providing possession."
      },
      {
        "section_id": "section_003",
        "section_title": "Authority Observations",
        "section_description": "MahaRERA has reviewed the case over several hearings, ultimately granting the complainants a refund of the full amount paid, along with interest under Section 18. The complaint has been referred to the Adjudicating Officer for the determination of compensation due to the complainants for violations of RERA."
      },
      {
        "section_id": "section_004",
        "section_title": "Decision",
        "section_description": "MahaRERA ordered the respondent to refund the amount with interest, directing that the matter be further reviewed by the Adjudicating Officer to determine compensation, due to the ongoing legal complexities and the jurisdictional authority issues."
      }
    ],
    "strong_points": [
      "MahaRERA’s clear order for a refund of the amount paid with interest.",
      "The determination that the issue of compensation is within the jurisdiction of the Adjudicating Officer ensures further legal clarity."
    ],
    "weak_points": [
      "The process of determining compensation is delayed due to jurisdictional questions.",
      "The prolonged legal process may delay the complainants' receipt of full compensation."
    ]
  },
  {
    "case_id": "case_047",
    "case_title": "Complaint No. CC006000000054965 - Sonal Ashish Gupta vs Ravi Developments",
    "case_description": "Sonal Ashish Gupta filed a complaint seeking possession of an apartment in the project 'Gaurav Excellency' by Ravi Developments, alleging a significant delay in the possession of the apartment. The respondent proposed an alternative apartment and settlement, but the complainant insisted on interest for the delay. MahaRERA is working to resolve the dispute, with the respondent required to obtain an occupancy certificate and hand over possession of the apartment as soon as possible.",
    "sections": [
      {
        "section_id": "section_001",
        "section_title": "Complainant’s Claims",
        "section_description": "The complainant has claimed that there was a significant delay in handing over the possession of her apartment in the 'Gaurav Excellency' project, despite a registered agreement for sale. She has demanded that the respondent either provide possession of the original apartment or compensation for the delay, including interest starting from October 2017."
      },
      {
        "section_id": "section_002",
        "section_title": "Respondent’s Status",
        "section_description": "The respondent submitted that the delay in the project was primarily due to delays in receiving necessary approvals from the competent authorities. They have proposed offering an alternative apartment and have executed a deed of rectification shifting the complainant's allotment to another apartment in the same project."
      },
      {
        "section_id": "section_003",
        "section_title": "Authority Observations",
        "section_description": "MahaRERA noted the arguments presented by both parties and acknowledged that the delay in project completion was largely due to external factors beyond the respondent's control. However, the complainant maintained that the delay was intentional, seeking interest compensation for the delay. MahaRERA directed the respondent to expedite the obtaining of the occupancy certificate and possession of the original apartment."
      },
      {
        "section_id": "section_004",
        "section_title": "Decision",
        "section_description": "MahaRERA directed the respondent to obtain the occupancy certificate as per the registration webpage and hand over possession of the original apartment at the earliest. The complainant's request for interest on the delayed possession will be adjudicated once the project nears completion, in accordance with Section 18 of the Real Estate (Regulation and Development) Act, 2016."
      }
    ],
    "strong_points": [
      "The complainant's claim is well-supported by a registered agreement for sale and documented delays in possession.",
      "The respondent has attempted to resolve the dispute by offering an alternative apartment and executing a deed of rectification."
    ],
    "weak_points": [
      "The respondent's delay in obtaining approvals has significantly affected the timeline of the project, causing frustration for the complainant.",
      "The complainant's claim for interest will be adjudicated only once the project nears completion, leaving a significant delay in resolution."
    ]
  },
  {
    "case_id": "case_048",
    "case_title": "Complaint No. CC006000000044189 - Subdey Jaipal Singh Chawhan vs Ekta Parksville Homes Pvt. Ltd.",
    "case_description": "Subdey Jaipal Singh Chawhan filed a complaint seeking compensation and a refund of amounts paid for an apartment in the 'Ekta Parksville Phase III' project by Ekta Parksville Homes Pvt. Ltd. The complainant alleged that the respondent failed to deliver the apartment and did not pay the pre-EMI. Despite several attempts at amicable settlement, the respondent has not met the original timeline. The case was transferred to the Adjudicating Officer for adjudication of compensation.",
    "sections": [
      {
        "section_id": "section_001",
        "section_title": "Complainant’s Claims",
        "section_description": "The complainant claimed that the respondent has failed to hand over possession of the apartment, and the pre-EMI payments have not been made. He requested to withdraw from the project and demanded a refund of the amounts paid along with interest and compensation for the delay."
      },
      {
        "section_id": "section_002",
        "section_title": "Respondent’s Status",
        "section_description": "The respondent expressed willingness to pay the pre-EMI but requested more time to settle the matter amicably. However, they acknowledged that the project completion has been delayed and cited the revised extended completion date of June 30, 2021, for the project."
      },
      {
        "section_id": "section_003",
        "section_title": "Authority Observations",
        "section_description": "MahaRERA noted that the respondent had been directed to complete the project by October 31, 2018, under a previous order. Despite this, the respondent has failed to hand over possession as directed. The complainant has now decided to withdraw from the project due to the ongoing delay."
      },
      {
        "section_id": "section_004",
        "section_title": "Decision",
        "section_description": "MahaRERA transferred the case to the Adjudicating Officer for adjudication of compensation, following the complainant's decision to withdraw from the project due to the respondent's failure to deliver the apartment within the stipulated time frame."
      }
    ],
    "strong_points": [
      "The complainant has clearly documented the delays and failure to meet the terms of the agreement, including non-payment of pre-EMI.",
      "The case was transferred to the Adjudicating Officer for compensation, which ensures further attention to the complainant's grievances."
    ],
    "weak_points": [
      "The respondent's failure to meet the original deadline and subsequent delays have significantly impacted the complainant's ability to benefit from the project.",
      "The case now rests with the Adjudicating Officer, which could further delay resolution and compensation for the complainant."
    ]
  },
  {
    "case_id": "case_049",
    "case_title": "Complaint No. CC005000000022746 - Pramod Kumar, Ashutosh Kumar Rai, Piyush Rajaram Sonawane, Tushar Rajaram Sonawane, Pramod Malhari Teli, Gaurav Pranay vs Arush Associates",
    "case_description": "A group of complainants filed a complaint against Arush Associates for not complying with a previous MahaRERA order to hand over possession of apartments by December 2018. Despite being given an extended deadline of December 2019, the respondent failed to complete the project. As a result, the complainants requested the revocation of the project’s registration and permission to carry out the remaining development work under Section 8 of the Real Estate (Regulation and Development) Act, 2016. MahaRERA revoked the registration of the project, citing non-completion and default under Section 7(1)(a) of the Act.",
    "sections": [
      {
        "section_id": "section_001",
        "section_title": "Complainants’ Claims",
        "section_description": "The complainants sought action against the respondent for failing to comply with a prior MahaRERA order requiring the handover of apartments. They requested the revocation of the project’s registration and that the association of allottees be allowed to complete the development work."
      },
      {
        "section_id": "section_002",
        "section_title": "Respondent’s Status",
        "section_description": "The respondent failed to complete the project by the deadline of December 2018, and despite a subsequent extension to December 2019, the project remained incomplete. No one appeared on behalf of the respondent during the final hearing."
      },
      {
        "section_id": "section_003",
        "section_title": "Authority Observations",
        "section_description": "MahaRERA acknowledged the failure of the respondent to complete the project as per the committed timelines and noted that the project had been sold out. The Authority invoked Section 7 of the Real Estate Act for the revocation of the project registration and allowed the association of allottees to take over the remaining development work."
      },
      {
        "section_id": "section_004",
        "section_title": "Decision",
        "section_description": "MahaRERA revoked the registration of the project as per Section 7 of the Real Estate Act, and directed the freezing of the project’s bank account. The association of allottees was granted the authority to carry out the remaining development work, subject to forming a legal entity."
      }
    ],
    "strong_points": [
      "MahaRERA effectively took action under Section 7(1)(a) for the non-completion of the project, revoking the registration to protect the interests of the complainants.",
      "The complainants took proactive steps by forming an association and presenting a detailed financial plan to complete the project."
    ],
    "weak_points": [
      "The respondent failed to appear during hearings and did not fulfill its obligations, delaying the project’s completion.",
      "The remaining development work will be dependent on the formation of a legal entity by the association of allottees, which may take time."
    ]
  },
  {
    "case_id": "case_050",
    "case_title": "Complaint No. CC005000000064828 - Ashwini Jadhav & Amrish Jadhav vs Suyog Structures (Partnership)",
    "case_description": "Ashwini Jadhav and Amrish Jadhav filed a complaint seeking the execution of an 'Agreement to Sale' for Flat No. A-501 in the 'Gulmohar Primrose' project. They also seek compensation for mental agony, legal expenses, and increased stamp duty due to the fraudulent conduct of the respondent, Suyog Structures (Partnership), who had failed to honor their agreement as per the booking application. The complaint also includes a dispute over the booking amount and dishonored payments.",
    "sections": [
      {
        "section_id": "section_001",
        "section_title": "Complainant’s Claims",
        "section_description": "The complainants sought the execution of the 'Agreement to Sale' for Flat No. A-501 in the 'Gulmohar Primrose' project, along with compensation for mental agony, legal expenses, and the potential increase in stamp duty due to changes in government policies. They also requested a refund of the booking amount along with interest due to the fraudulent actions of the respondent."
      },
      {
        "section_id": "section_002",
        "section_title": "Respondent’s Status",
        "section_description": "The respondent, Suyog Structures (Partnership), argued that the complainants had paid only part of the booking amount, with a dishonored cheque. They stated that the complainants were unable to meet the full payment and had used unlawful methods to pressure the respondent. The respondent also claimed the project had been completed and the occupancy certificate had been issued prior to the complainants' booking."
      },
      {
        "section_id": "section_003",
        "section_title": "Authority Observations",
        "section_description": "MahaRERA noted that the Occupancy Certificate (OC) for the project was issued before the complainants' booking, and the agreement for sale had not been executed due to issues with the complainants' payment. MahaRERA acknowledged the need for a detailed hearing on the merits of the case, with both parties required to submit convenience documents as per the MahaRERA Order No. 27 of 2021."
      },
      {
        "section_id": "section_004",
        "section_title": "Decision",
        "section_description": "MahaRERA concluded that no urgency had been demonstrated by the complainants. The complaint would be listed according to its seniority and tagged with other complaints related to the same project. MahaRERA decided to hear the case in detail after both parties submitted their respective documents."
      }
    ],
    "strong_points": [
      "The complainants have clearly outlined their claims for relief, including the execution of the agreement for sale and compensation for various losses.",
      "MahaRERA is ensuring that both parties submit all necessary documentation for a fair and detailed hearing."
    ],
    "weak_points": [
      "The complaint is likely to be delayed as it will be heard based on seniority and not as an urgent matter.",
      "The dispute over the booking amount and dishonored payments could complicate the resolution of the case."
    ]
  },
  {
    "case_id": "case_051",
    "case_title": "Complaint No. CC0050000002172 - Ravindra Sadashiv Rane vs M/s. Ashirwad Associates and others",
    "case_description": "Ravindra Sadashiv Rane and Mr. Raju Moreshwar Mudhol filed complaints seeking a refund of the amounts paid for booking flats in the 'Ashirwad Samruddhi' project by M/s. Ashirwad Associates under Section 18 of the RERA, citing failure of possession delivery within the stipulated time. The MahaRERA noted that the respondents failed to hand over possession as per the agreement, and the complaints were clubbed together for hearing.",
    "sections": [
      {
        "section_id": "section_001",
        "section_title": "Complainant’s Claims",
        "section_description": "The complainants, Ravindra Sadashiv Rane and Mr. Raju Moreshwar Mudhol, seek a refund of the amounts paid for the booking of their flats in the 'Ashirwad Samruddhi' project, under Section 18 of RERA. The complainants argue that possession of their flats, as per the agreements for sale, was not delivered within the stipulated period of December 2016."
      },
      {
        "section_id": "section_002",
        "section_title": "Respondent’s Status",
        "section_description": "The respondents, M/s. Ashirwad Associates, argued that there was no privity of contract between them and the complainants, and claimed that the complainants were not entitled to file the complaints. However, the complainants had entered into registered agreements for sale with the respondents."
      },
      {
        "section_id": "section_003",
        "section_title": "Authority Observations",
        "section_description": "MahaRERA acknowledged that the complainants were allottees under the respondents' project and had entered into registered agreements for sale. The authority noted that the agreed possession dates had already passed, and the complainants sought a refund along with interest and compensation as per Section 18 of RERA."
      },
      {
        "section_id": "section_004",
        "section_title": "Decision",
        "section_description": "MahaRERA transferred the complaints to the Adjudicating Officer in Pune for appropriate action under RERA. The respondents were granted the liberty to agitate their contentions before the Adjudicating Officer, with all issues remaining open for consideration."
      }
    ],
    "strong_points": [
      "The complainants have registered agreements for sale and are entitled to seek relief under Section 18 of RERA due to delayed possession.",
      "MahaRERA acknowledged the failure of possession delivery by the respondents, supporting the complainants' claims for a refund."
    ],
    "weak_points": [
      "The respondents contested the complaints, arguing no privity of contract and lack of locus standi for the complainants.",
      "The final decision on the complainants' claims is deferred to the Adjudicating Officer, which could delay relief for the complainants."
    ]
  } 
]

df = pd.DataFrame(data)

# Load InLegalBERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("law-ai/InLegalBERT")

# Define PyTorch Dataset
class LegalDataset(Dataset):
    def __init__(self, cases, tokenizer):
        self.cases = cases
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases.iloc[idx]
        inputs = self.tokenizer(
            case["case_description"], truncation=True, padding="max_length", max_length=512, return_tensors="pt"
        )
        labels = torch.tensor([0])  # Placeholder label, adjust as needed
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
df = df.reset_index(drop=True)  # Ensure proper indexing

for train_index, test_index in kf.split(df):
    train_data, test_data = df.iloc[train_index], df.iloc[test_index]

    train_dataset = LegalDataset(train_data, tokenizer)
    test_dataset = LegalDataset(test_data, tokenizer)

    # Load InLegalBERT Model
    model = BertForSequenceClassification.from_pretrained("law-ai/InLegalBERT", num_labels=len(df["sections"].iloc[0]))

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./legalis_model")
tokenizer.save_pretrained("./legalis_model")

# Compute Relevancy Score using Cosine Similarity
def get_relevance_score(input_text, model, tokenizer):
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        input_embedding = model(**inputs).logits.numpy()

    section_embeddings = []
    section_titles = []
    for section in df["sections"].iloc[0]:
        section_inputs = tokenizer(section["section_description"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        with torch.no_grad():
            section_embedding = model(**section_inputs).logits.numpy()
        section_embeddings.append(section_embedding)
        section_titles.append(section["section_title"])

    scores = cosine_similarity(input_embedding, np.array(section_embeddings).squeeze())
    normalized_scores = softmax(scores.flatten())  # Apply softmax for normalization
    
    sorted_results = sorted(zip(section_titles, normalized_scores), key=lambda x: x[1], reverse=True)
    return sorted_results
