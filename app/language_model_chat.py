# --- START OF FILE app/language_model_chat.py ---
import streamlit as st
import os
# Импорт на константи от config.py
from config import LLM_MODEL_DIR, LLM_MODEL_NAME

# Уверете се, че llama_cpp е инсталиран: pip install llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Конструиране на пътя до модела.
# main.py се стартира от директорията *над* app/,
# така че пътят оттам до модела ще бъде 'app/models/MODEL_NAME'
RELATIVE_MODEL_PATH_FROM_ROOT = os.path.join(
    "app", LLM_MODEL_DIR, LLM_MODEL_NAME)


def check_model_availability():
    """Проверява дали моделът съществува на очаквания път."""
    if not os.path.exists(RELATIVE_MODEL_PATH_FROM_ROOT):
        st.error(
            f"❌ Моделът '{LLM_MODEL_NAME}' липсва! Очаква се да бъде в '{RELATIVE_MODEL_PATH_FROM_ROOT}'. "
            f"Свали го от: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf "
            f"и го постави в поддиректория '{os.path.join('app', LLM_MODEL_DIR)}'."
        )
        return False
    return True


@st.cache_resource
def load_llm_model():
    """Зарежда и кешира LLM модела."""
    st.info(
        f"Зареждане на LLM модел '{LLM_MODEL_NAME}'... Това може да отнеме известно време.")
    try:
        llm = Llama(
            model_path=RELATIVE_MODEL_PATH_FROM_ROOT,
            n_ctx=1024,
            # ЗАДАДЕНО НА 0 ЗА CPU ПО ПОДРАЗБИРАНЕ. Променете ако имате GPU и конфигуриран llama-cpp-python.
            n_gpu_layers=0,
            n_threads=6,
            verbose=False
        )
        st.success(f"LLM модел '{LLM_MODEL_NAME}' е зареден успешно.")
        return llm
    except Exception as e:
        st.error(f"Грешка при зареждане на LLM модела: {e}")
        st.error(
            "Уверете се, че имате инсталиран `llama-cpp-python` (`pip install llama-cpp-python`).")
        st.error(
            "Ако използвате GPU (n_gpu_layers > 0), уверете се, че компилацията на llama-cpp-python е с GPU поддръжка.")
        return None


# Преименувах prompt на current_user_prompt за яснота
def generate_llm_response(llm_instance, current_user_prompt):
    """Генерира отговор от LLM въз основа на историята на чата и текущия промпт."""
    if llm_instance is None:
        st.error("LLM моделът не е зареден, не мога да генерирам отговор.")
        return None
    try:
        # Изграждане на историята на чата за модела от st.session_state
        # st.session_state[session_messages_key] вече съдържа последния user_prompt
        chat_history_for_model = []
        # Ключ, използван в render_page
        session_messages_key = f"llm_messages_{LLM_MODEL_NAME}"

        # Включваме всички съобщения до момента, включително последния user_prompt
        for msg in st.session_state.get(session_messages_key, []):
            chat_history_for_model.append(
                {"role": msg["role"], "content": msg["content"]})

        if not chat_history_for_model or chat_history_for_model[-1]["role"] != "user":
            # Това не би трябвало да се случи, ако логиката в render_page е правилна
            st.warning(
                "Историята на чата изглежда непълна преди извикване на LLM.")
            return None

        st.info("🤖 Генериране на отговор от LLM...")
        response = llm_instance.create_chat_completion(
            messages=chat_history_for_model,
            temperature=0.7,
            max_tokens=350,  # Увеличено малко
            stream=False
        )

        if response and 'choices' in response and len(response['choices']) > 0:
            ai_content = response['choices'][0]['message']['content']
            return ai_content.strip()
        else:
            st.warning("LLM не върна очакван отговор.")
            # st.json(response) # За дебъг, ако response не е каквото очакваме
            return None
    except Exception as e:
        st.error(f"Грешка по време на генериране на отговор от LLM: {e}")
        # import traceback
        # st.error(traceback.format_exc()) # За по-детайлен traceback при дебъг
        return None


def render_page():
    # По-кратко име на модела за заглавие
    st.header(f"💬 Чат с {LLM_MODEL_NAME.split('-')[0]}")

    if not LLAMA_CPP_AVAILABLE:
        st.error(
            "Библиотеката `llama-cpp-python` не е намерена. Моля, инсталирайте я: `pip install llama-cpp-python`")
        st.stop()

    if not check_model_availability():
        st.stop()

    llm_instance = load_llm_model()

    if llm_instance is None:
        st.error(
            "LLM Моделът не можа да бъде зареден. Чат функционалността не е достъпна.")
        st.stop()

    session_messages_key = f"llm_messages_{LLM_MODEL_NAME}"
    if session_messages_key not in st.session_state:
        st.session_state[session_messages_key] = []
        # Примерно първо съобщение от асистента, ако е нужно
        # st.session_state[session_messages_key].append(
        #     {"role": "assistant", "content": "Здравейте! Как мога да ви помогна днес?"}
        # )

    # Показване на съществуващите съобщения
    for message in st.session_state[session_messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Напишете вашето съобщение..."):
        st.session_state[session_messages_key].append(
            {"role": "user", "content": user_prompt})
        with st.chat_message("user"):  # Показваме съобщението на потребителя веднага
            st.markdown(user_prompt)

        # Подготвяме място за отговора на асистента
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Плейсхолдър за стрийминг или за "мисля..."
            message_placeholder.markdown("🤖 Мисля...")

            ai_response_content = generate_llm_response(
                llm_instance, user_prompt)

            if ai_response_content:
                message_placeholder.markdown(
                    ai_response_content)  # Показваме отговора
                st.session_state[session_messages_key].append(
                    {"role": "assistant", "content": ai_response_content})
            else:
                message_placeholder.warning(
                    "Не получих отговор от асистента или възникна грешка.")
                # Ако искаме да премахнем последния user prompt при грешка:
                # if st.session_state[session_messages_key][-1]["role"] == "user":
                #     st.session_state[session_messages_key].pop()

        # st.rerun() # Обикновено не е нужен с st.chat_input, тъй като той предизвиква презареждане.
        # Ако има проблеми със синхронизацията на състоянието, може да се наложи.
