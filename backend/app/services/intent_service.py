from collections import OrderedDict
import re
from typing import List


INTENT_LABELS = {
    "disease_desc": "查询疾病简介",
    "disease_cause": "查询疾病病因",
    "disease_prevent": "查询疾病预防措施",
    "disease_cycle": "查询疾病治疗周期",
    "disease_prob": "查询治愈概率",
    "disease_population": "查询疾病易感人群",
    "disease_drugs": "查询疾病所需药品",
    "disease_do_eat": "查询疾病宜吃食物",
    "disease_not_eat": "查询疾病忌吃食物",
    "disease_check": "查询疾病所需检查项目",
    "disease_department": "查询疾病所属科目",
    "disease_symptom": "查询疾病的症状",
    "disease_cure_way": "查询疾病的治疗方法",
    "disease_acompany": "查询疾病的并发疾病",
    "drug_producer": "查询药品的生产商",
}


class IntentRuleEngine:
    def __init__(self) -> None:
        self.intent_keywords = OrderedDict(
            {
                "disease_cause": ["病因", "原因", "为什么", "怎么得", "引起"],
                "disease_prevent": ["预防", "避免", "防止"],
                "disease_cycle": ["治疗周期", "多久", "多长时间", "几天", "几周", "几个月"],
                "disease_prob": ["治愈概率", "治愈率", "能治好吗", "好不好治"],
                "disease_population": ["易感人群", "哪些人容易", "高发人群", "什么人容易"],
                "disease_drugs": ["药", "用药", "吃什么药", "药品"],
                "disease_do_eat": ["宜吃", "吃什么", "饮食建议", "推荐食物"],
                "disease_not_eat": ["忌吃", "不能吃", "禁忌食物"],
                "disease_check": ["检查", "查什么", "怎么确诊", "检测"],
                "disease_department": ["科室", "挂号", "挂什么科", "哪个科"],
                "disease_symptom": ["症状", "表现", "现象"],
                "disease_cure_way": ["治疗", "怎么治", "治疗方法", "怎么办"],
                "disease_acompany": ["并发", "并发症", "一起出现"],
                "drug_producer": ["生产商", "厂家", "哪个公司", "谁生产"],
                "disease_desc": ["简介", "是什么", "介绍", "概述"],
            }
        )
        self.symptom_consult_re = re.compile(
            r"(疼|痛|胀|晕|吐|恶心|腹泻|拉肚子|发烧|发热|咳|痒|麻|不舒服)"
        )

    def detect(self, query: str) -> List[str]:
        text = query.strip()
        if not text:
            return []
        result: List[str] = []
        for intent, keywords in self.intent_keywords.items():
            if any(k in text for k in keywords):
                result.append(intent)

        if not result and self.symptom_consult_re.search(text):
            result.append("disease_cure_way")
        if not result:
            result.append("disease_desc")
        return result[:5]
