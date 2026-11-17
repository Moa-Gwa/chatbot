# Moa-Gwa Chatbot

가톨릭대학교 규정, 부서 연락처를 근거로 답하는 RAG 챗봇 서비스.
학생 요청은 BE(로그인 확인, 요청 한도)를 거쳐 이 서비스의 REST API로 전달된다. 클러스터 내부에서만 열린다.

- Python, FastAPI, LangGraph
- 임베딩, 답변 생성: Gemini

자세한 구조와 실행 방법은 구현과 함께 정리한다.
