import google.generativeai as genai
import os
import sys

# 💡 이 변수는 사용자가 실제 API 키를 넣거나 환경 변수를 설정해야 합니다.
# 캔버스 환경이 아닌 일반 콘솔 앱을 가정하고 코드를 구성합니다.
# 만약 환경 변수가 설정되어 있지 않다면, 여기에 실제 키를 넣으세요. (현재는 빈 문자열 유지)
my_gemini_api_key = "" 

def main():
    """
    Gemini API를 사용하여 파일 업로드(이미지, PDF 등)가 가능한
    대화형 탄소 배출 분석 전문가 챗봇을 실행합니다.
    """
    
    # 1. API 키 설정
    # 환경 변수 'GEMINI_API_KEY'를 먼저 확인하고, 없으면 my_gemini_api_key를 사용합니다.
    api_key = os.environ.get("GEMINI_API_KEY", my_gemini_api_key)

    if not api_key:
        print("환경 변수 'GEMINI_API_KEY'가 설정되지 않았습니다.")
        print("코드의 'my_gemini_api_key' 변수에 실제 API 키를 넣어주시거나 환경 변수를 설정해야 합니다.")
        return

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"[오류] API 키 설정에 실패했습니다: {e}")
        return

    # 2. 모델 및 시스템 명령어 설정 (✨ 탄소 배출 전문가 페르소나 정의)
    expert_system_instruction = (
        "당신은 탄소 배출 및 환경 분석 전문가입니다. "
        "사용자가 업로드한 파일(예: PDF)의 내용을 분석하여 질문에 깊이 있고 정확하게 답하세요. "
        "전문가로서의 신뢰감을 주는 어조(존댓말)를 사용하고, 답변 시 관련된 사실과 수치, 혹은 문서의 핵심 내용을 명확히 제시해 주세요. "
        "만약 답변할 수 있는 정보가 부족하다면, '제공된 문서 내에서는 해당 정보를 찾을 수 없습니다.'라고 정중하게 말해주세요. "
        "모든 대답은 한국어로 하고, 답변 마지막에는 항상 🌿 이모지를 넣어주세요."
    )
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"[오류] 모델 로딩에 실패했습니다: {e}")
        return

    # 3. 대화 세션 시작 (시스템 명령어를 config에 포함)
    chat = model.start_chat(
        history=[],
        config={"system_instruction": expert_system_instruction} # 페르소나 적용
    )

    print("--- 🌿 탄소 배출 분석 전문가 챗봇 (파일 업로드 가능) ---")
    print("안녕하세요! 저는 당신의 탄소 배출 전문가입니다. '탄소 분석.pdf'와 같은 파일을 첨부하여 질문해 주세요.")
    print("'그만'을 입력하면 종료됩니다.")
    print("-" * 50)

    # 4. 대화 루프
    while True:
        uploaded_file = None # 매 턴마다 업로드된 파일 객체 초기화
        
        # 4-1. 파일 경로 입력 받기
        try:
            file_path = input("📎 업로드할 파일 경로 (없으면 Enter): ").strip()
        except EOFError:
            break
        
        # 파일 경로가 있다면 파일 업로드 시도
        if file_path:
            print(f"파일 업로드 중... ({file_path})")
            try:
                # 파일을 API에 업로드하고 파일 객체를 받습니다.
                uploaded_file = genai.upload_file(path=file_path)
                print(f"✅ 파일 업로드 성공! 파일명: {uploaded_file.display_name}")
            except FileNotFoundError:
                print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")
                continue # 다음 루프로 이동
            except Exception as e:
                print(f"[오류] 파일 업로드에 실패했습니다: {e}")
                print("지원되는 파일 형식(JPG, PNG, PDF 등)인지 확인하세요.")
                continue # 다음 루프로 이동

        # 4-2. 사용자 텍스트 입력 받기
        try:
            if uploaded_file:
                user_input = input("You (파일에 대해 질문): ")
            else:
                user_input = input("You (텍스트로 질문): ")
        except EOFError:
            break
            
        # 4-3. 종료 조건 확인
        if user_input.lower() == '그만':
            print("Gemini Expert: 🤖 대화를 종료합니다. 탄소 중립 목표 달성을 응원합니다. 🌿")
            break

        if not user_input.strip(): # 빈 입력은 무시
            if uploaded_file:
                # 텍스트 없이 파일만 업로드된 경우, 파일 객체만 삭제 후 계속
                print(f"🗑️ 업로드된 파일 삭제 중...")
                uploaded_file.delete()
            continue

        # 4-4. 파일과 텍스트를 함께 전송
        content_to_send = []
        content_to_send.append(user_input)

        # (중요) 이번 턴에 업로드된 파일이 있다면, 리스트에 추가합니다.
        if uploaded_file:
            content_to_send.append(uploaded_file)

        # 4-5. 채팅 세션에 [텍스트] 또는 [텍스트, 파일] 리스트 전송
        try:
            response_stream = chat.send_message(content_to_send, stream=True)
            print("Gemini Expert: 🤖 ", end="")

            # 스트리밍 응답 출력
            for chunk in response_stream:
                print(chunk.text, end="", flush=True)

            print() # 응답 완료 후 줄바꿈
        
        except Exception as e:
            print(f"\n\n[오류 발생]: {e}")
            print("API 요청 중 문제가 발생했습니다. 입력을 다시 시도해주세요.")
            
        finally:
            # 4-6. (⭐ 중요) 파일 객체 사용 후 반드시 삭제합니다.
            if uploaded_file:
                # 파일을 삭제하는 동안 콘솔에 안내 메시지를 출력합니다.
                sys.stderr.write("🗑️ 업로드된 파일 삭제 중...\n")
                uploaded_file.delete()
                sys.stderr.write("🗑️ 삭제 완료.\n")

if __name__ == "__main__":
    main()
