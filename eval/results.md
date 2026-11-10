# Eval results

18 cases x 3 runs per config. LLM `llama3`, embeddings `nomic-embed-text`, 300-char chunks, rag keeps 50% of chunks.

| Config | Avg context chars | Score bias | MAE vs label | Correlation | Rec. accuracy (model) | Rec. accuracy (from score) | Run-to-run stdev | Failed calls | Sec per LLM call |
|---|---|---|---|---|---|---|---|---|---|
| rag:v1 | 969 | +13.9 | 15.9 | 0.91 | 54% | 61% | 6.8 | 0 | 34.7 |
| full:v1 | 1596 | +15.2 | 17.8 | 0.88 | 50% | 54% | 4.3 | 0 | 32.5 |
| rag:v2 | 969 | +6.6 | 10.9 | 0.93 | 70% | 63% | 6.9 | 0 | 60.3 |

## Per case (mean score, model's first-run recommendation)

| Resume | JD | Label | rag:v1 | full:v1 | rag:v2 |
|---|---|---|---|---|---|
| priya_backend | backend_python | 90 Hire | 88 Hire | 87 Hire | 92 Hire |
| priya_backend | frontend_react | 15 No Hire | 27 Interview | 27 Review | 32 No Hire |
| priya_backend | ml_engineer | 30 No Hire | 73 Interview | 70 Interview | 64 Review |
| arjun_ml | ml_engineer | 92 Hire | 88 Hire | 88 Hire | 88 Interview |
| arjun_ml | backend_python | 45 Review | 67 Interview | 77 Interview | 55 Review |
| arjun_ml | data_analyst | 40 Review | 57 Interview | 60 Interview | 27 Review |
| lena_frontend | frontend_react | 93 Hire | 82 Interview | 85 Hire | 88 Hire |
| lena_frontend | backend_python | 10 No Hire | 30 No Hire | 22 No Hire | 3 No Hire |
| maria_analyst | data_analyst | 90 Hire | 89 Hire | 82 Hire | 94 Hire |
| maria_analyst | ml_engineer | 10 No Hire | 47 Interview | 55 Interview | 35 Review |
| sam_bootcamp | frontend_react | 35 Review | 53 Interview | 60 Interview | 50 Review |
| sam_bootcamp | backend_python | 15 No Hire | 20 Review | 22 No Hire | 19 No Hire |
| tom_devops | backend_python | 55 Interview | 78 Interview | 85 Interview | 66 Interview |
| tom_devops | ml_engineer | 30 No Hire | 43 No Hire | 43 Interview | 22 No Hire |
| wei_datascience | ml_engineer | 55 Interview | 80 Hire | 78 Interview | 77 Interview |
| wei_datascience | data_analyst | 70 Interview | 82 Interview | 78 Interview | 84 Interview |
| hannah_nurse | data_analyst | 5 No Hire | 13 No Hire | 20 Review | 3 No Hire |
| hannah_nurse | frontend_react | 0 No Hire | 13 No Hire | 15 No Hire | 0 No Hire |
