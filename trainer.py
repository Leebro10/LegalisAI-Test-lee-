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
          "section_title": "Section 13 of the RERA Act, 2016: No Advance Without Agreement for Sale",
          "section_description": "A promoter shall not accept more than 10% of the cost of the apartment, plot, or building as an advance or application fee without first entering into a written agreement for sale with the buyer. This agreement must specify development details, timelines, and the possession date."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "Promoters must ensure transparency by disclosing all relevant project details, such as sanctioned plans, layout plans, cost estimates, and timelines. They must also ensure that advertisements and agreements align with the registered project details."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 18 of the RERA Act, 2016: Right to Compensation for Delays",
          "section_description": "If the promoter fails to deliver the project as per the terms of the agreement for sale, the buyer is entitled to withdraw from the project with a full refund and interest. Alternatively, the buyer may continue with the project and receive interest for the delay until possession is granted."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 12 of the RERA Act, 2016: Compensation for False or Misleading Advertisements",
          "section_description": "If a buyer suffers loss due to false or misleading information in the advertisement or prospectus, they are entitled to compensation. The promoter is liable to return the amount paid by the buyer along with interest."
        }
      ],
      "strong_points": [
        "Complainant raised concerns about ambiguities in the draft agreement and discrepancies from the cost sheet.",
        "Complainant’s willingness to pay the remaining amount and proceed with the project."
      ],
      "weak_points": [
        "Complainant has paid only 4% of the total consideration, falling short of the RERA requirement.",
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
          "section_title": "Section 18 of the RERA Act, 2016: Compensation for Delays in Possession",
          "section_description": "If a promoter fails to deliver possession of a property as per the terms of the agreement, the buyer is entitled to withdraw and claim a refund with interest. Alternatively, the buyer may choose to remain in the project and seek interest for the delay until possession is handed over."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 32(g) of the RERA Act, 2016: Promoting Conciliation",
          "section_description": "MahaRERA has the responsibility to facilitate amicable resolution of disputes between promoters and buyers through its Conciliation Forum, promoting a fair settlement process."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 4(2)(l)(D) of the RERA Act, 2016: Adherence to Timelines",
          "section_description": "The promoter must adhere to the timelines and specifications mentioned in the agreement for sale. In case of non-compliance, buyers have the right to seek remedies under RERA."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Authority’s Powers for Adjudication",
          "section_description": "MahaRERA has the power to hear disputes and ensure compliance with the provisions of the Act. Conciliation and adjudication are part of this process, and parties may resolve issues amicably under the Authority's guidance."
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
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "Promoters are required to provide accurate and transparent information regarding the status of inventory, including sold and available units, on the MahaRERA website."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 34 of the RERA Act, 2016: Functions of the Authority",
          "section_description": "The Authority is responsible for maintaining an updated database of registered projects, including details of sold and available inventory, to ensure transparency and protect buyers' interests."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 31 of the RERA Act, 2016: Filing of Complaints",
          "section_description": "Any aggrieved person can file a complaint with MahaRERA if they believe there is a contravention of the Act or Rules, such as incorrect or missing inventory details."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Authority’s Powers for Adjudication",
          "section_description": "MahaRERA has the power to resolve disputes and verify complaints, including ensuring the accuracy of project details, such as inventory listings."
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
          "section_title": "Section 18 of the RERA Act, 2016: Compensation for Delays in Possession",
          "section_description": "If a promoter fails to deliver possession of a property as per the terms of the agreement, the buyer is entitled to withdraw and claim a refund with interest. Alternatively, the buyer may continue with the project and seek interest for the delay until possession is handed over."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 72 of the Insolvency and Bankruptcy Code, 2016: Claims Against Insolvent Entities",
          "section_description": "If a respondent is undergoing insolvency proceedings, any claims for compensation or unpaid dues must be filed with the appointed Resolution Professional overseeing the insolvency process."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 34 of the RERA Act, 2016: Functions of the Authority",
          "section_description": "MahaRERA has the responsibility to ensure compliance with its orders and assist complainants in resolving disputes, including directing them to alternative legal remedies when necessary."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Authority’s Powers for Adjudication",
          "section_description": "MahaRERA is empowered to hear complaints and applications, such as non-execution filings, but its jurisdiction is limited when insolvency proceedings are involved."
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
          "section_title": "Section 18 of the RERA Act, 2016: Compensation for Delays in Possession",
          "section_description": "If the promoter fails to deliver possession by the agreed date, the allottee is entitled to withdraw from the project and claim a refund with interest or continue with the project and claim compensation for the delay."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 71 of the RERA Act, 2016: Adjudication of Compensation Claims",
          "section_description": "The Adjudicating Officer is empowered to decide on claims for compensation under Section 18, ensuring fair resolution of disputes."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11(4)(a) of the RERA Act, 2016: Promoter's Obligations for Timely Possession",
          "section_description": "Promoters are obligated to adhere to the terms of the agreement for sale, including delivering possession within the stipulated timeline."
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
          "section_title": "Section 18 of the RERA Act, 2016: Compensation for Delays in Possession",
          "section_description": "If the promoter fails to deliver possession by the agreed date, or fails to obtain the occupancy certificate, the allottee is entitled to withdraw and claim a refund with interest or continue with the project and seek compensation for delays or lapses."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 71 of the RERA Act, 2016: Adjudication of Compensation Claims",
          "section_description": "The Adjudicating Officer is empowered to decide the quantum of compensation payable to the complainant in cases involving violations under Section 18."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11(4)(a) of the RERA Act, 2016: Promoter’s Obligations",
          "section_description": "The promoter is responsible for obtaining the occupancy certificate and adhering to the terms of the agreement for sale, ensuring the timely and lawful transfer of the property to the allottee."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 19(10) of the RERA Act, 2016: Obligations of the Allottee",
          "section_description": "An allottee is obligated to make payments and take possession of the property as agreed. However, they retain the right to seek compensation if the promoter fails to comply with statutory obligations, such as obtaining the occupancy certificate."
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
          "section_title": "Section 19(6) of the RERA Act, 2016: Allottee's Obligation to Pay Dues",
          "section_description": "An allottee must make all payments for the property as per the agreement for sale, including any installments. Failure to pay dues can result in legal claims from the promoter along with interest for the delay."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 72 of the RERA Act, 2016: Factors for Determining Compensation",
          "section_description": "The Adjudicating Officer must consider the nature of default, mental agony, and the extent of loss caused to the complainant when determining the quantum of compensation."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11(4)(a) of the RERA Act, 2016: Promoter's Obligations for Timely Recovery",
          "section_description": "Promoters are required to ensure compliance with agreements, including the timely recovery of dues. They are entitled to seek legal remedies if allottee obligations are unmet."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 3 of the MOFA Act, 1963: Disclosure Obligations",
          "section_description": "Under the Maharashtra Ownership of Flats Act, promoters are required to disclose all details of agreements and financial obligations clearly. This reinforces transparency in payment disputes."
        },
        {
          "section_id": "section_005",
          "section_title": "Section 71 of the RERA Act, 2016: Adjudication of Compensation Claims",
          "section_description": "The Adjudicating Officer is empowered to hear and decide compensation claims related to violations under RERA, including disputes about unpaid dues and related harassment claims."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund for Non-completion or Delays",
          "section_description": "If the promoter fails to complete or deliver possession of a property as agreed, the allottee is entitled to a refund of the amount paid along with interest. Alternatively, the allottee can claim compensation for delays while continuing with the project."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the Insolvency and Bankruptcy Code, 2016: Moratorium",
          "section_description": "Once a company is admitted into Corporate Insolvency Resolution Process (CIRP), a moratorium is imposed prohibiting the initiation or continuation of legal proceedings against the debtor until the process is resolved."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 19(6) of the RERA Act, 2016: Allottee’s Rights in Case of Non-compliance",
          "section_description": "The allottee has the right to demand a refund or compensation for violations under the terms of the agreement, especially in cases of delays or failure to deliver the project."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Adjudication of Compensation Claims",
          "section_description": "The Adjudicating Officer is empowered to decide on claims related to compensation under RERA. However, jurisdiction is subject to restrictions like the moratorium under IBC."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If the promoter fails to hand over possession by the agreed date, the allottee is entitled to withdraw from the project and claim a refund along with interest and compensation. Alternatively, the allottee may continue and seek compensation for delays."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 71 of the RERA Act, 2016: Adjudication of Compensation Claims",
          "section_description": "The Adjudicating Officer is empowered to decide the quantum of compensation payable to the allottee in cases involving delays or breaches under Section 18."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11(4)(a) of the RERA Act, 2016: Promoter’s Obligations",
          "section_description": "Promoters are obligated to adhere to the terms of the agreement for sale, including delivering possession within the stipulated timeline. Failure to comply constitutes a breach under RERA."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 19(10) of the RERA Act, 2016: Allottee’s Rights in Case of Delay",
          "section_description": "The allottee is entitled to seek remedies for delays or breaches by the promoter, including refunds, interest, and compensation as specified under RERA."
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
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "Promoters must fulfill statutory obligations, provide accurate disclosures, and ensure the timely completion of the project as per the agreement for sale."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If the promoter fails to deliver possession by the agreed date, the allottee or society of allottees is entitled to a refund or compensation for the delay, along with interest."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 19(3) of the RERA Act, 2016: Rights of the Allottee",
          "section_description": "The allottee has the right to claim possession of the property as per the terms of the agreement for sale and the timelines specified therein."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must execute the project in accordance with the sanctioned plans and cannot make any alterations or delays without the consent of the allottees."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If the promoter fails to deliver possession by the agreed date, the allottee is entitled to withdraw from the project and claim a refund with interest or continue and seek compensation for the delay."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 19(3) of the RERA Act, 2016: Allottee’s Right to Possession",
          "section_description": "The allottee has the right to claim possession of the property as per the terms of the agreement for sale and within the specified timeline."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must execute the project as per the sanctioned plans and agreements, without undue delay or deviations, except with the consent of the allottees."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 4 of the MOFA Act, 1963: Promoter’s Obligations",
          "section_description": "Promoters are obligated to complete the construction and provide possession as per the terms agreed upon in the agreement for sale."
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
          "section_title": "Section 12 of the RERA Act, 2016: Refund for False or Misleading Representations",
          "section_description": "If any promoter provides false or misleading information in advertisements or brochures, or fails to comply with commitments, the allottee is entitled to a refund of the amount paid along with interest and compensation."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If the promoter fails to complete or deliver possession as per the agreement, the allottee may withdraw from the project and claim a refund with interest or choose to continue and seek compensation for the delay."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 13 of the RERA Act, 2016: Advance Payment and Agreement for Sale",
          "section_description": "A promoter cannot accept more than 10% of the cost of the property as an advance or application fee without first executing a registered agreement for sale with the allottee."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 4 of the MOFA Act, 1963: Promoter’s Obligations",
          "section_description": "Promoters are obligated to execute agreements and provide possession of the property as per the terms agreed upon, ensuring transparency and adherence to timelines."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If the promoter fails to complete or deliver possession by the agreed date, the allottee may withdraw from the project and claim a refund with interest. Alternatively, the allottee may continue and seek compensation for the delay."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 13 of the RERA Act, 2016: Advance Payment and Agreement for Sale",
          "section_description": "A promoter cannot accept more than 10% of the cost of the property as an advance or application fee without first executing a registered agreement for sale with the allottee."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "Promoters are obligated to provide accurate disclosures and ensure adherence to statutory obligations, including timely completion of the project and possession as agreed."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 4 of the MOFA Act, 1963: Promoter’s Obligations",
          "section_description": "Promoters must fulfill their obligations under the agreement for sale, including the timely handover of possession as per the agreed terms."
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
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "The promoter is required to adhere to approved project plans and specifications, ensure timely possession, and execute agreements for sale for buyers who have paid over 10% of the property price. The failure to complete the project in a timely manner makes the promoter liable for compensation or refunds with interest."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 15 of the RERA Act, 2016: Transfer of Real Estate Project",
          "section_description": "This section specifies that the promoter cannot transfer the majority rights and liabilities in a project without the prior written consent of at least two-thirds of the allottees and approval from MahaRERA. This is directly relevant to the transfer of the project to the new promoter."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 19 of the RERA Act, 2016: Rights and Duties of Allottees",
          "section_description": "Allottees have the right to receive timely possession of their apartments and demand refunds or compensation for delays. They also have the right to participate in the formation of an association or society to safeguard their interests."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "The promoter cannot accept more than 10% of the apartment cost as an advance without executing a registered agreement for sale. This is relevant for buyers seeking agreements from the promoter."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If the promoter fails to complete the project or is unable to give possession of the property in accordance with the terms of the agreement, the allottee is entitled to a refund of the amount paid along with interest and compensation as determined by the Authority."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "The promoter is prohibited from accepting more than 10% of the cost of the apartment as an advance without executing a registered agreement for sale. This is relevant since the respondents failed to execute the agreement despite receiving payment."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to adjudicate on matters concerning compensation and interest payable to allottees for violations by the promoter under RERA."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "This section entitles allottees to seek a refund, compensation, and interest if the promoter fails to complete the project or deliver possession in accordance with the agreement. This applies to the delayed possession of flats Nos. 603 and 604."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 12 of the RERA Act, 2016: Obligations Regarding False Representation",
          "section_description": "If any false representation is made about the project, allottees are entitled to seek compensation for the loss or mental harassment caused. This aligns with the complainant's claim of misrepresentation about the possession date."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "The promoter must ensure that the project adheres to the approved plans, specifications, and timelines. Any deviation entitles the allottee to seek compensation or rectification."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to decide on compensation and interest payable to allottees for violations under RERA. It is relevant as the case has been referred to the Adjudicating Officer for a decision."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "If the promoter fails to complete the project or hand over possession within the agreed timeline, the allottee is entitled to seek a refund of the amount paid, along with interest and compensation. This applies directly to the complainant's claim for refund and compensation."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 6 of the RERA Act, 2016: Extension of Registration Due to Force Majeure",
          "section_description": "This section allows for the extension of the project registration in case of force majeure events, such as the COVID-19 pandemic. The respondents cited this provision to justify the delay in possession."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "Promoters cannot accept more than 10% of the property price without executing a registered agreement for sale. Since the possession timeline was breached despite a registered agreement, this section reinforces the complainant's right to compensation."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to adjudicate compensation claims. The case being referred for determining the compensation amount is consistent with this provision."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to a refund of the amount paid, along with interest and compensation, if the promoter fails to complete the project or hand over possession as per the agreed timeline. The complainants sought relief under this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "Promoters cannot accept more than 10% of the property cost as an advance without executing a registered agreement for sale. This reinforces the complainants' rights under the agreement."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 14 of the Insolvency and Bankruptcy Code, 2016: Moratorium During CIRP",
          "section_description": "This section imposes a moratorium during the Corporate Insolvency Resolution Process (CIRP), prohibiting the continuation of legal proceedings against the corporate debtor. This provision was applied by MahaRERA to adjourn the case."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 7 of the MOFA Act, 1963: Adherence to Approved Plans",
          "section_description": "The promoter is obligated to adhere to the approved plans and timelines specified in the agreement. Any deviation or delay may entitle the allottee to seek remedies under the law."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to seek a refund, along with interest and compensation, in case of delayed possession or non-compliance with the terms of the agreement. It supports the complainant’s claims for rent compensation and interest."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 11(4)(e) of the RERA Act, 2016: Formation of Allottee Association",
          "section_description": "Promoters are required to facilitate the formation of an allottee association. Non-compliance with agreements or settlements made with the association can lead to further legal obligations for the promoter."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "Promoters cannot collect more than 10% of the property cost without a registered agreement for sale. The complainant’s claim for compliance with settlement terms aligns with obligations under this section."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 7 of the MOFA Act, 1963: Adherence to Approved Plans",
          "section_description": "The promoter is obligated to adhere to the approved plans and timelines as per agreements. Any deviation entitles the allottee to seek remedies, including compensation for delays or non-compliance."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to seek a refund, compensation, and interest if the promoter fails to deliver possession or provide agreed amenities within the stipulated timeline. The complainant’s claim for delayed possession, non-provision of amenities, and lack of an occupancy certificate falls under this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must adhere to sanctioned plans and specifications, including promised amenities like schools, hospitals, and lifts. Failure to provide these facilities constitutes a breach of the agreement."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 31 of the RERA Act, 2016: Filing of Complaints with MahaRERA",
          "section_description": "This section empowers allottees to file complaints against promoters for non-compliance with RERA provisions. It underpins the complainant's right to file and pursue their case."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section authorizes the Adjudicating Officer to adjudicate matters related to compensation and interest. The case being transferred to the Adjudicating Officer aligns with this section."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "If the promoter fails to deliver possession as per the agreement, the allottee is entitled to seek a refund or claim compensation with interest for the delay. This section directly addresses the complainants' claim of delayed possession by 1885 days."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "The promoter is prohibited from collecting more than 10% of the total cost without executing a registered agreement for sale. The supplementary agreement cited by the respondents to extend the possession timeline is relevant under this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must adhere to approved project plans and timelines. Deviations, such as delayed possession, entitle allottees to remedies, including compensation or rectification."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to decide on compensation and interest. The case being transferred to the Adjudicating Officer aligns with this provision."
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
          "section_title": "Section 31 of the RERA Act, 2016: Filing of Complaints with MahaRERA",
          "section_description": "This section empowers any aggrieved person to file a complaint regarding violations of RERA provisions or rules. The complainant invoked this section to challenge the construction on land designated for common amenities."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must adhere to the sanctioned plans and project specifications. Any deviation, such as constructing buildings on land meant for common amenities, constitutes a violation of this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 7 of the MOFA Act, 1963: Adherence to Approved Plans",
          "section_description": "This section mandates that promoters adhere to approved layouts and plans as per agreements. The complainant’s objection to using FSI for unauthorized construction aligns with this provision."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 3 of the RERA Act, 2016: Prior Registration of Real Estate Projects",
          "section_description": "All ongoing and new real estate projects must be registered with MahaRERA. If the disputed buildings were not part of the registered project plan, this section could apply to assess compliance."
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
          "section_title": "Section 12 of the RERA Act, 2016: Obligations Regarding False Representation",
          "section_description": "If a promoter provides false or misleading information or violates the terms of an agreement, the allottee is entitled to compensation or a refund with interest. This section applies to the complainants' claim that the respondent violated the terms of mutual understanding."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to a refund of the amount paid, along with interest and compensation, if the promoter fails to comply with the terms of an agreement or mutual understanding."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to decide on matters related to compensation and refunds. The respondent's contention that the case should be filed with the Adjudicating Officer rather than MahaRERA aligns with this provision."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 4 of the MOFA Act, 1963: Obligation to Execute Agreement for Sale",
          "section_description": "Promoters are obligated to execute a clear and legally binding agreement for sale. Any deviations or disputes arising from this agreement may result in claims for compensation or refunds."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "If the promoter fails to deliver possession in accordance with the terms of the agreement, the allottee is entitled to claim interest or a refund with compensation. This section directly addresses the complainant’s claim for delayed possession and interest."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters are obligated to complete the project as per sanctioned plans and timelines, including obtaining necessary approvals like the occupancy certificate. Failure to do so constitutes a breach of this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 19 of the RERA Act, 2016: Rights and Duties of Allottees",
          "section_description": "Allottees have the right to demand possession as per the agreement, along with essential amenities and required approvals. They are also entitled to claim interest or compensation for delays caused by the promoter."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 7 of the MOFA Act, 1963: Adherence to Approved Plans",
          "section_description": "Under MOFA, promoters must adhere to the approved plans and timelines agreed upon with the allottees. The lack of an occupancy certificate and incomplete amenities breaches this obligation."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "If the promoter fails to complete the project or hand over possession as per the agreement, the allottee is entitled to a refund along with interest and compensation. This section forms the basis of the complainant’s demand for a refund and interest."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "Promoters cannot accept more than 10% of the property cost as an advance without executing a registered agreement for sale. The arbitration clause within the agreement becomes relevant under this provision."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to decide matters related to compensation, including refunds and interest. It is relevant to the complainant’s demand for financial remedies."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 8 of the Arbitration and Conciliation Act, 1996",
          "section_description": "This section requires disputes to be referred to arbitration if an arbitration clause exists in the agreement. The respondent’s defense relies on this provision, questioning the maintainability of the complaint."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "If the promoter fails to deliver possession or fulfill the terms of the agreement, the allottee is entitled to seek a refund with interest and compensation. The complainants relied on this section for their refund and compensation claim."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the Insolvency and Bankruptcy Code, 2016: Moratorium During CIRP",
          "section_description": "This section imposes a moratorium during the Corporate Insolvency Resolution Process (CIRP), preventing legal proceedings against the corporate debtor. MahaRERA deferred its decision under the moratorium imposed by the NCLT."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to decide on compensation and refunds. However, the complaint was deferred as the NCLT moratorium restricted proceedings."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 7 of the MOFA Act, 1963: Adherence to Approved Plans",
          "section_description": "The promoter is obligated to adhere to approved plans and timelines under MOFA. The complainants' claim for refunds is indirectly tied to the promoter's failure to meet agreed deadlines, but this remains subordinate to the insolvency process."
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
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "This section prohibits promoters from accepting more than 10% of the total cost of the apartment without executing a registered agreement for sale. The complainant's demand for executing an agreement for sale aligns with this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section allows the allottee to seek compensation, interest, or a refund if the promoter fails to deliver possession as per the agreement. It is directly applicable to the complainant's claim for delayed possession and compensation."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 12 of the RERA Act, 2016: Obligations Regarding False Representation",
          "section_description": "If the promoter provides false or misleading information, the allottee is entitled to compensation for the loss incurred. This applies to the complainant's claim of false information and the unauthorized sale of a flat to a third party."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "This section mandates that promoters adhere to sanctioned plans and prohibits changes without the consent of two-thirds of the allottees. The complainant's grievance regarding amendments to plans without consent falls under this section."
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
          "section_title": "Section 4 of the RERA Act, 2016: Application for Registration of Real Estate Projects",
          "section_description": "Promoters are required to submit project-related documents, including sanctioned plans, layout approvals, and timelines for project completion, as part of the registration process. The complainant sought compliance with this section for the 'Shristi Sea View' project."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "This section obligates promoters to provide accurate project details, update information regularly, and ensure transparency in project status. The complainant's request for document uploads aligns with these obligations."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 14 of the Insolvency and Bankruptcy Code, 2016: Moratorium During CIRP",
          "section_description": "This section imposes a moratorium during the Corporate Insolvency Resolution Process (CIRP), preventing legal proceedings against the corporate debtor. MahaRERA adjourned the case sine die due to the moratorium imposed by the NCLT."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to address violations and determine compliance with RERA provisions. However, due to the moratorium, the proceedings were paused pending the resolution of insolvency proceedings."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section allows allottees to claim a refund, interest, or compensation if the promoter fails to deliver possession as per the agreement. The complainant's request for a refund and compensation is directly based on this provision."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 12 of the RERA Act, 2016: Obligations Regarding False Representation",
          "section_description": "If a promoter provides misleading or false information about the project, the allottee is entitled to compensation for the resulting loss. The complainant's allegation of misleading information regarding the possession date aligns with this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to adjudicate claims for refunds and compensation. The transfer of the case to the Adjudicating Officer is based on this provision."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 44 of the RERA Act, 2016: Appeal to the Appellate Tribunal",
          "section_description": "This section governs appeals against orders passed by MahaRERA or the Adjudicating Officer. The ongoing appeal challenging the interim order made this section relevant and led to the case's adjournment."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to claim compensation, interest, or a refund if the promoter fails to deliver possession within the stipulated time. The complainants' request for compensation for delayed possession aligns with this provision."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters are required to adhere to approved plans and specifications. The complainants' allegations of BMC norm violations and carpet area deficiencies relate to this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "This section mandates promoters to complete the project as per the approved timeline and provide regular updates. The requirement to obtain the occupancy certificate (OC) and complete remaining work aligns with the promoter's obligations under this section."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 19 of the RERA Act, 2016: Rights and Duties of Allottees",
          "section_description": "Allottees have the right to demand possession along with essential documentation like the occupancy certificate. The complainants’ request to ensure compliance with BMC norms and obtain the OC aligns with this section."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to claim possession, interest, or compensation if the promoter fails to deliver possession by the agreed date. The complainants' claim for possession, interest, and compensation directly falls under this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "This section mandates that promoters must adhere to sanctioned plans and specifications. The complainants' concerns about construction defects and incomplete promised amenities are addressed under this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "Promoters are obligated to complete the project as per the approved timeline and provide all required approvals, including occupancy certificates. The respondents' delay and partial compliance align with issues addressed under this section."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to adjudicate claims for compensation. The transfer of the complaint to the Adjudicating Officer aligns with this provision."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section allows allottees to seek a refund, interest, and compensation if the promoter fails to deliver possession as per the agreement or comply with orders issued by MahaRERA. The complainant’s claim is directly based on this provision."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 40 of the RERA Act, 2016: Recovery of Interest or Penalty or Compensation",
          "section_description": "This section provides mechanisms to recover compensation, interest, or penalties imposed under RERA. It supports the complainant’s non-execution application, as the respondent failed to comply with an earlier MahaRERA order."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "Promoters must adhere to timelines and ensure compliance with orders issued by MahaRERA. The respondent’s failure to complete the work within the directed timeline breaches this obligation."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section empowers the Adjudicating Officer to determine the quantum of compensation in cases involving refunds and financial claims. The case transfer aligns with this provision."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to withdraw from a project if the promoter fails to complete or deliver possession as per the terms of the agreement. The complainants' claim for a refund and interest due to the project's standstill aligns with this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 7 of the RERA Act, 2016: Revocation of Registration",
          "section_description": "If the promoter fails to comply with the conditions of registration or provisions of RERA, the project's registration may be revoked. The complainants’ reference to the non-extension of the project’s registration highlights the relevance of this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "This section requires promoters to ensure that the project is completed on time and to update project details on the MahaRERA portal. The project's standstill and lack of updates indicate a breach of this obligation."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 38 of the RERA Act, 2016: Powers of MahaRERA to Issue Interim Orders",
          "section_description": "This section empowers MahaRERA to pass interim orders, such as granting the respondent one final opportunity to reply while adhering to the principles of natural justice. The authority's actions are consistent with this provision."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section allows allottees to claim interest, compensation, or a refund if the promoter fails to deliver possession as per the terms of the agreement. The complainants' claims for delayed possession align with this provision."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 7 of the RERA Act, 2016: Revocation of Registration",
          "section_description": "If a promoter fails to comply with RERA provisions or the terms of registration, the authority may revoke the project’s registration. This section is relevant as the project’s registration expired on December 31, 2019, without renewal."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 8 of the RERA Act, 2016: Obligation After Revocation of Registration",
          "section_description": "This section outlines the procedure for completing a project after the revocation of its registration. MahaRERA's directive for forming an Association of Allottees (AOA) aligns with this section, enabling the AOA to take over the project’s completion."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 11 of the RERA Act, 2016: Obligations of the Promoter",
          "section_description": "This section mandates the promoter to adhere to timelines, provide project updates, and comply with statutory obligations. The respondent’s failure to upload the Form 5 Annual Audit Report and obtain the Occupancy Certificate falls under this section."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section allows allottees to claim possession, interest, or compensation if the promoter fails to deliver possession as per the agreement. The complainants' request for possession and compensation for delayed possession aligns with this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 7 of the RERA Act, 2016: Revocation of Registration",
          "section_description": "If a promoter abandons a project, as admitted by the respondent, this section empowers MahaRERA to revoke the project’s registration and initiate steps to protect the interests of allottees."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 8 of the RERA Act, 2016: Obligation After Revocation of Registration",
          "section_description": "This section outlines the steps to ensure project completion or refund after the revocation of registration. The complainants' insistence on possession can align with the provisions for project completion."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 71 of the RERA Act, 2016: Power of Adjudicating Officer",
          "section_description": "This section grants the Adjudicating Officer authority to determine the quantum of compensation and interest, making it relevant to the transfer of the complaint for further adjudication."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to claim refunds, interest, or compensation if possession is delayed. This is the core provision for the complainants' claims."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 40 of the RERA Act, 2016: Recovery of Interest or Penalty or Compensation",
          "section_description": "This section enables the recovery of compensation or interest due to the allottee, serving as a mechanism for enforcing the Adjudicating Officer's decision."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11(4)(a) of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must develop the project in accordance with sanctioned plans and approved specifications. The lack of an occupancy certificate violates this provision."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 72 of the RERA Act, 2016: Factors to Determine Compensation",
          "section_description": "This section outlines the factors the Adjudicating Officer must consider while determining compensation, including the promoter’s non-compliance and impact on the allottee."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section allows allottees to claim a refund or possession with interest if the promoter fails to deliver possession as per the agreement. The complainants initially sought a refund under this section but later attempted to amend their relief to possession."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must adhere to sanctioned plans and project timelines. The failure to deliver possession on time aligns with the violation of this section, supporting the complainants’ initial claim for a refund."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 34(f) of the RERA Act, 2016: Functions of MahaRERA",
          "section_description": "This section empowers MahaRERA to ensure compliance with its orders, including adjudicating disputes and addressing concerns like project delays or the extension of registration. The directive for the respondents to seek a project extension aligns with this section."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 40 of the RERA Act, 2016: Recovery of Interest or Penalty or Compensation",
          "section_description": "This section provides a mechanism for enforcing the recovery of awarded refunds or interest. It supports the complainants’ entitlement to a refund and interest under the original reliefs."
        }
      ],
      "strong_points": [
        "The complainants are persistent in their pursuit of justice, having approached MahaRERA and MahaREAT for a resolution of their grievances.",
        "MahaRERA adhered to the principles of law, emphasizing the need for efficiency in the litigation process and a final resolution."
      ],
      "weak_points": [
        "The change in relief sought by the complainants after several years of litigation raised concerns about the abuse of legal provisions to seek a more financially beneficial outcome.",
        "The case faced delays due to the lengthy litigation process, complicating the decision-making and causing extended uncertainty for the complainants."
      ]
    },
    {
      "case_id": "case_038",
      "case_title": "Complaint No. CC006000000100497 - Mrs. Roopa N. Hegde and Mr. Narahari V. Hegde vs M/s. Sanvo Resort Pvt. Ltd",
      "case_description": "Mrs. Roopa N. Hegde and Mr. Narahari V. Hegde filed a complaint seeking possession of flat No. 2005, 20th floor, S-1 Wing-B, in the project 'Marathon Nexzone Zenith' along with interest and compensation for delayed possession under Section 18 of the Real Estate (Regulation and Development) Act, 2016 (RERA). The case was heard through video conferencing, and both parties presented their submissions. The complainants requested the case be transferred to the Ld. Adjudicating Officer/MahaRERA for adjudging compensation. The respondent had no objection to this transfer. The case has been transferred for further proceedings as per the provisions of RERA.",
      "sections": [
        {
          "section_id": "section_001",
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "Allottees are entitled to claim possession with interest for delayed possession or a refund with compensation if the promoter fails to adhere to the agreed possession date."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 71 of the RERA Act, 2016: Adjudication of Compensation Claims",
          "section_description": "The Adjudicating Officer is empowered to hear and decide compensation claims, ensuring that allottees receive appropriate remedies under RERA."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11(4)(b) of the RERA Act, 2016: Promoter's Responsibility to Obtain Approvals",
          "section_description": "Promoters are required to secure necessary approvals and permits to complete the project and deliver possession as per the agreement."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 35 of the RERA Act, 2016: Power to Call for Information",
          "section_description": "MahaRERA has the authority to call for information, documents, or evidence from promoters to ensure compliance and resolve disputes."
        },
        {
          "section_id": "section_005",
          "section_title": "Section 8 of the MOFA Act, 1963: Obligations in Case of Delay",
          "section_description": "If possession is not delivered as per the agreement, the promoter must refund the amounts received with interest and compensate the allottee for any financial loss."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If the promoter fails to complete or deliver possession by the agreed date, the allottee may claim possession with interest for the delay or withdraw and seek a refund with compensation."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 32(g) of the RERA Act, 2016: Promoting Conciliation",
          "section_description": "MahaRERA is tasked with facilitating amicable resolution of disputes through its Conciliation Forum to promote harmony between promoters and allottees."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 40 of the RERA Act, 2016: Recovery of Interest or Penalty",
          "section_description": "If consent terms or other obligations are not fulfilled, the recovery of interest, penalty, or compensation can be enforced as an arrear of land revenue."
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
          "section_title": "Section 6 of the RERA Act, 2016: Extension of Registration",
          "section_description": "Promoters can seek an extension of project registration in case of unforeseen and unavoidable circumstances. However, such extensions should not impact the rights of the allottees."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 19(4) of the RERA Act, 2016: Allottee’s Right to Demand Timely Possession",
          "section_description": "Allottees have the right to demand possession of their property as per the agreed timeline. Delays without valid justification entitle the allottee to remedies under RERA."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 34(b) of the RERA Act, 2016: Functions of MahaRERA to Ensure Timely Completion",
          "section_description": "MahaRERA is tasked with ensuring timely completion and delivery of projects by monitoring project progress and addressing complaints."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 7 of the RERA Act, 2016: Revocation of Registration",
          "section_description": "If a promoter fails to comply with obligations, MahaRERA can revoke the project’s registration, allowing allottees to seek alternative remedies or project completion."
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
          "section_title": "Section 18(1) of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "Allottees are entitled to a refund or possession with interest for delayed possession. This includes compensation for financial and mental hardship caused by delays."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 18(3) of the RERA Act, 2016: Determination of Compensation",
          "section_description": "The Adjudicating Officer is empowered to decide the quantum of compensation for delayed possession, factoring in financial and other losses suffered by the allottee."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 15 of the RERA Act, 2016: Transfer of Real Estate Projects",
          "section_description": "Promoters cannot transfer or assign a project to a third party without prior written consent of two-thirds of the allottees and MahaRERA's approval. This ensures continuity and protection of allottees' rights."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 35 of the RERA Act, 2016: Power to Call for Information",
          "section_description": "MahaRERA can summon the promoter to provide details about the project transfer and any agreements impacting the rights of the allottees."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If a promoter fails to deliver possession as per the agreement, the allottee is entitled to withdraw from the project and claim a refund with interest. Alternatively, the allottee may continue and seek compensation for the delay."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 44 of the RERA Act, 2016: Appeals to Real Estate Appellate Tribunal",
          "section_description": "Any party aggrieved by a MahaRERA order may appeal to the Real Estate Appellate Tribunal, which has the authority to review and remand cases for fresh adjudication."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 38 of the RERA Act, 2016: Powers of MahaRERA",
          "section_description": "MahaRERA has the authority to pass interim orders, enforce rulings, and ensure compliance with the provisions of RERA in resolving disputes."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters are required to execute projects in compliance with the sanctioned plans and agreements, ensuring no deviations that may impact allottees' rights."
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
          "section_title": "Section 13 of the RERA Act, 2016: Advance Payment and Agreement for Sale",
          "section_description": "Promoters cannot accept more than 10% of the total cost of the property as an advance without executing a registered agreement for sale. This ensures legal clarity and safeguards the rights of allottees."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 6 of the RERA Act, 2016: Extension of Registration",
          "section_description": "If a project’s registration lapses, the promoter must seek an extension from MahaRERA, provided the delay is due to valid and unavoidable circumstances."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 7 of the RERA Act, 2016: Revocation of Registration",
          "section_description": "MahaRERA can revoke a project’s registration if the promoter fails to comply with statutory obligations or acts against the interests of allottees, ensuring accountability."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 8 of the RERA Act, 2016: Completion of Project by New Promoter",
          "section_description": "If a project’s registration is revoked, MahaRERA can appoint a new promoter or association of allottees to complete the project, ensuring the interests of buyers are protected."
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
          "section_title": "Section 11(5) of the RERA Act, 2016: Obligations of Promoters Regarding Agreements",
          "section_description": "Promoters must ensure that agreements for sale comply with the provisions of RERA. Clauses that contravene the Act or deny buyers their statutory rights may be deemed invalid."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 31 of the RERA Act, 2016: Filing of Complaints",
          "section_description": "Allottees have the right to file complaints with MahaRERA for violations of the Act, irrespective of clauses like arbitration agreements that may attempt to limit their legal recourse."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 44 of the RERA Act, 2016: Appeals to Real Estate Appellate Tribunal",
          "section_description": "MahaRERA can defer a decision if a similar matter is under consideration by the Real Estate Appellate Tribunal, ensuring consistent rulings and judicial precedence."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must adhere to the sanctioned plans and agreements for sale, ensuring compliance with statutory obligations and buyer rights."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If possession is delayed beyond the agreed date, the allottee is entitled to claim possession with interest for the delay or seek a refund with compensation, depending on their choice."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 6 of the RERA Act, 2016: Extension of Registration",
          "section_description": "Promoters can request an extension of the project registration in the event of force majeure circumstances, ensuring legitimate delays are accounted for without compromising buyer rights."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 37 of the RERA Act, 2016: General Powers of MahaRERA",
          "section_description": "MahaRERA has the authority to issue directions and consolidate related complaints for efficient resolution, ensuring fairness and comprehensive adjudication."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 19(4) of the RERA Act, 2016: Rights of Allottees",
          "section_description": "Allottees have the right to demand timely possession and seek remedies for delays, including interest and compensation, if the promoter fails to meet their obligations."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If possession is delayed beyond the agreed date, the allottee may claim a refund with interest or continue with the project and seek compensation for the delay."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 12 of the RERA Act, 2016: False or Misleading Representations",
          "section_description": "If a promoter provides false or misleading information, the allottee is entitled to claim a refund with interest and compensation for any losses incurred."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 13 of the RERA Act, 2016: Advance Payment and Agreement for Sale",
          "section_description": "Promoters cannot accept more than 10% of the property cost as an advance without first executing a registered agreement for sale with the allottee."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 31 of the RERA Act, 2016: Filing of Complaints",
          "section_description": "Any aggrieved person may file a complaint with MahaRERA regarding violations of the Act, ensuring their rights are upheld through legal recourse."
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
          "section_title": "Section 18 of the RERA Act, 2016: Refund and Compensation for Delays",
          "section_description": "If possession is delayed beyond the agreed date, the allottee can claim interest for the delay or withdraw from the project with a refund and compensation."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must deliver possession as per the sanctioned plans and agreements. Any deviation or delay requires the consent of the allottee and appropriate compensation."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 19(4) of the RERA Act, 2016: Allottee’s Right to Timely Possession",
          "section_description": "Allottees are entitled to possession of their property within the timeline specified in the agreement and can seek remedies for delays caused by the promoter."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 40 of the RERA Act, 2016: Enforcement of Orders",
          "section_description": "MahaRERA has the authority to enforce its orders, including directions for possession, compensation, and obtaining occupancy certificates by the promoter."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to withdraw from the project and claim a refund with interest and compensation if the promoter fails to deliver possession as per the agreement. The complainant’s decision to withdraw and seek compensation aligns with this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "Promoters cannot collect more than 10% of the property cost without executing a registered agreement for sale. If the respondent collected funds without fulfilling this obligation, it constitutes a breach of this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 40 of the RERA Act, 2016: Recovery of Interest or Penalty or Compensation",
          "section_description": "This section provides mechanisms for recovering compensation or penalties awarded under RERA. It supports the complainant’s claim for compensation for delays and non-payment of pre-EMI."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 8 of the RERA Act, 2016: Obligation After Revocation of Registration",
          "section_description": "If the project is abandoned or significantly delayed, this section outlines the promoter’s responsibility to refund payments or allow the allottees to take over the project through an Association of Allottees (AOA). The complainant’s withdrawal from the project highlights the applicability of this section."
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
          "section_title": "Section 7(1)(a) of the RERA Act, 2016: Revocation of Registration",
          "section_description": "This section empowers MahaRERA to revoke the registration of a project if the promoter violates terms of the registration, fails to complete the project, or engages in unfair practices. This section was invoked to revoke the respondent’s project registration."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 8 of the RERA Act, 2016: Obligation After Revocation of Registration",
          "section_description": "This section outlines the process for completing a project after its registration has been revoked. The complainants' request to allow the association of allottees to take over the development aligns with this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 11(4)(d) of the RERA Act, 2016: Obligation to Deposit Funds in a Separate Account",
          "section_description": "Promoters are required to deposit 70% of the project funds into a separate account to ensure project completion. MahaRERA's directive to freeze the project's bank account ties to the enforcement of this obligation."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 34(f) of the RERA Act, 2016: Functions of MahaRERA",
          "section_description": "This section mandates MahaRERA to take measures ensuring compliance with its orders. The decision to revoke the project registration and empower the association of allottees reflects this role."
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
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "Promoters cannot collect more than 10% of the apartment’s cost without executing a registered agreement for sale. The complainants’ claim for the execution of an agreement for sale directly relates to this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 12 of the RERA Act, 2016: Obligations Regarding False Representation",
          "section_description": "If the promoter provides misleading information or breaches commitments, the allottee can claim compensation. The complainants’ allegations of fraudulent conduct by the respondent align with this section."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 14 of the RERA Act, 2016: Adherence to Sanctioned Plans",
          "section_description": "Promoters must adhere to project plans and agreements as per the approved layout. The dispute over the booking application and non-execution of the agreement ties to this obligation."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 35 of the RERA Act, 2016: Powers to Call for Information",
          "section_description": "MahaRERA can require both parties to submit documents or information for a detailed hearing. The authority’s directive to submit convenience documents is supported by this section."
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
          "section_title": "Section 18 of the RERA Act, 2016: Return of Amount and Compensation",
          "section_description": "This section entitles allottees to claim refunds, interest, or compensation if the promoter fails to deliver possession as per the agreement. The complainants' claims for a refund align with this section."
        },
        {
          "section_id": "section_002",
          "section_title": "Section 13 of the RERA Act, 2016: No Deposit or Advance Without Agreement for Sale",
          "section_description": "Promoters cannot collect more than 10% of the property cost without executing a registered agreement for sale. The complainants’ reliance on their registered agreements ties this section to the case."
        },
        {
          "section_id": "section_003",
          "section_title": "Section 72 of the RERA Act, 2016: Factors to Determine Compensation",
          "section_description": "This section specifies the factors to be considered by the Adjudicating Officer while determining compensation, such as the promoter’s conduct and the extent of loss caused to the complainants."
        },
        {
          "section_id": "section_004",
          "section_title": "Section 35 of the RERA Act, 2016: Powers to Call for Information",
          "section_description": "This section empowers MahaRERA to call for documents and explanations from the respondents to verify the complainants’ claims, ensuring procedural transparency during the hearing."
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
