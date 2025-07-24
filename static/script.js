// Aguarda o DOM carregar completamente antes de inicializar
document.addEventListener('DOMContentLoaded', () => {
    // Estabelece conexão WebSocket com o servidor
    const socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port);

    // Variáveis de estado da aplicação
    let currentConfig = {}; // Configurações atuais
    let isListening = false; // Status de escuta

    // Referências aos elementos da interface
    const settingsBtn = document.getElementById('settings-btn');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const cancelSettingsBtn = document.getElementById('cancel-settings-btn');
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const settingsModal = document.getElementById('settings-modal');
    
    // Campos do formulário de configurações
    const apiKeyInput = document.getElementById('api-key');
    const audioDeviceSelect = document.getElementById('audio-device-select');
    const numSuggestionsSelect = document.getElementById('num-suggestions');
    const answerDelayInput = document.getElementById('answer-delay');
    const languageSelect = document.getElementById('language-select');
    const disableTranslationCb = document.getElementById('disable-translation-cb');

    // Containers de exibição de conteúdo
    const sttContainer = document.getElementById('stt-container');
    const translationContainer = document.getElementById('translation-container');
    const suggestionsContainer = document.getElementById('suggestions-container');
    const translationWrapper = document.getElementById('translation-wrapper');
    const suggestionsWrapper = document.getElementById('suggestions-wrapper');

    // Indicadores de status
    const apiStatusIndicator = document.getElementById('api-status-indicator');
    const apiStatusText = document.getElementById('api-status-text');
    const notification = document.getElementById('notification');

    // Ícones SVG para os botões
    const playIcon = `<svg class="w-12 h-12" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"></path></svg>`;
    const pauseIcon = `<svg class="w-12 h-12" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clip-rule="evenodd"></path></svg>`;

    // Inicializa o botão com o ícone de play
    playPauseBtn.innerHTML = playIcon;

    // Funções para controle do modal de configurações
    const openModal = () => {
        settingsModal.classList.remove('hidden');
        settingsModal.classList.add('flex');
    };
    const closeModal = () => {
        settingsModal.classList.add('hidden');
        settingsModal.classList.remove('flex');
    };
    
    // Adiciona eventos aos botões do modal
    settingsBtn.addEventListener('click', openModal);
    closeModalBtn.addEventListener('click', closeModal);
    cancelSettingsBtn.addEventListener('click', closeModal);
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) closeModal();
    });

    /**
     * Atualiza a interface com as configurações recebidas
     * @param {Object} config - Objeto com as configurações
     */
    function updateUIFromConfig(config) {
        apiKeyInput.value = config.apiKey || '';
        audioDeviceSelect.value = config.audio_device_id || '';
        numSuggestionsSelect.value = config.num_suggestions || 3;
        answerDelayInput.value = config.answer_delay_ms || 500;
        languageSelect.value = config.target_language || 'en';
        disableTranslationCb.checked = config.disable_translation || false;
        
        // Controla visibilidade dos painéis baseado na configuração
        const isDisabled = disableTranslationCb.checked;
        translationWrapper.classList.toggle('hidden', isDisabled);
        suggestionsWrapper.classList.toggle('hidden', isDisabled);

        renderSuggestionBoxes(config.num_suggestions);
    }

    // Salva as configurações quando o botão é clicado
    saveSettingsBtn.addEventListener('click', () => {
        const configToSave = {
            apiKey: apiKeyInput.value,
            audio_device_id: audioDeviceSelect.value,
            target_language: languageSelect.value,
            answer_delay_ms: parseInt(answerDelayInput.value, 10),
            num_suggestions: parseInt(numSuggestionsSelect.value, 10),
            disable_translation: disableTranslationCb.checked
        };
        socket.emit('save_config', configToSave);
        closeModal();
    });

    // Atualiza interface quando checkbox de tradução é alterado
    disableTranslationCb.addEventListener('change', (e) => {
        const isDisabled = e.target.checked;
        translationWrapper.classList.toggle('hidden', isDisabled);
        suggestionsWrapper.classList.toggle('hidden', isDisabled);
        socket.emit('save_config', { disable_translation: isDisabled });
    });

    // Controla início/parada da escuta
    playPauseBtn.addEventListener('click', () => {
        if (isListening) {
            // Para a escuta se já estiver ativa
            socket.emit('stop_listening');
        } else {
            // Inicia a escuta com as configurações atuais
            const currentClientConfig = {
                apiKey: apiKeyInput.value,
                audio_device_id: audioDeviceSelect.value,
                target_language: languageSelect.value,
                answer_delay_ms: parseInt(answerDelayInput.value, 10),
                num_suggestions: parseInt(numSuggestionsSelect.value, 10),
                disable_translation: disableTranslationCb.checked
            };
            socket.emit('start_listening', currentClientConfig);
        }
    });

    /**
     * Exibe notificação temporária para o usuário
     * @param {string} message - Mensagem a ser exibida
     * @param {string} status - Tipo da notificação ('success' ou 'error')
     */
    function showNotification(message, status = 'success') {
        notification.textContent = message;
        notification.className = `fixed bottom-5 right-5 text-white py-2 px-4 rounded-lg shadow-lg transition-transform duration-300`;
        notification.classList.add(status === 'success' ? 'bg-green-600' : 'bg-red-600');
        notification.classList.remove('hidden', 'translate-y-20');
        setTimeout(() => {
            notification.classList.add('translate-y-20');
            setTimeout(() => notification.classList.add('hidden'), 300);
        }, 3000);
    }

    /**
     * Atualiza texto do elemento e rola para o final
     * @param {HTMLElement} element - Elemento a ser atualizado
     * @param {string} text - Texto a ser inserido
     */
    function updateAndScroll(element, text) {
        element.textContent = text;
        element.scrollTop = element.scrollHeight;
    }

    /**
     * Atualiza indicador de status da API
     * @param {string} status - Status da API ('active', 'inactive', 'error')
     * @param {string} message - Mensagem de status
     */
    function updateApiStatus(status, message) {
        apiStatusIndicator.className = 'w-3 h-3 rounded-full transition-colors duration-500';
        apiStatusIndicator.classList.add(`status-${status}`);
        apiStatusIndicator.title = `Status da API: ${message}`;
        apiStatusText.textContent = message;
    }

    /**
     * Renderiza caixas de sugestões vazias
     * @param {number} count - Número de caixas a renderizar
     */
    function renderSuggestionBoxes(count) {
        suggestionsContainer.innerHTML = '';
        for (let i = 0; i < count; i++) {
            const box = document.createElement('div');
            box.className = 'suggestion-box';
            suggestionsContainer.appendChild(box);
        }
    }

    /**
     * Atualiza o conteúdo das sugestões
     * @param {Array} suggestions - Array com textos das sugestões
     */
    function updateSuggestions(suggestions = []) {
        const boxes = suggestionsContainer.querySelectorAll('.suggestion-box');
        boxes.forEach((box, i) => {
            box.textContent = suggestions[i] || '';
        });
    }

    // Eventos do Socket.IO
    socket.on('connect', () => console.log('Conectado ao servidor!'));

    // Recebe atualizações de configuração do servidor
    socket.on('config_update', (config) => {
        console.log('Configurações recebidas do servidor:', config);
        currentConfig = config;
        updateUIFromConfig(config);
    });
    
    // Recebe lista de dispositivos de áudio disponíveis
    socket.on('audio_devices_list', (devices) => {
        console.log('Dispositivos de áudio recebidos:', devices);
        audioDeviceSelect.innerHTML = '';
        if (!devices || devices.length === 0) {
            audioDeviceSelect.innerHTML = '<option value="">Nenhum dispositivo encontrado</option>';
            return;
        }
        devices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.id;
            option.textContent = device.name;
            audioDeviceSelect.appendChild(option);
        });
        // Seleciona o dispositivo configurado anteriormente
        if (currentConfig.audio_device_id) {
            audioDeviceSelect.value = currentConfig.audio_device_id;
        }
    });

    // Atualiza interface quando a escuta é iniciada
    socket.on('listening_started', () => {
        isListening = true;
        playPauseBtn.innerHTML = pauseIcon;
        playPauseBtn.classList.add('listening');
    });

    // Atualiza interface quando a escuta é parada
    socket.on('listening_stopped', () => {
        isListening = false;
        playPauseBtn.innerHTML = playIcon;
        playPauseBtn.classList.remove('listening');
    });
    
    // Exibe notificações do servidor
    socket.on('notification', (data) => showNotification(data.message, data.status));
    
    // Atualiza status da API
    socket.on('api_status', (data) => updateApiStatus(data.status, data.message));

    // Atualiza conteúdo dos painéis em tempo real
    socket.on('update_stt', (data) => updateAndScroll(sttContainer, data.text));
    socket.on('update_translation', (data) => updateAndScroll(translationContainer, data.text));
    socket.on('update_suggestions', (data) => updateSuggestions(data.suggestions));
    
    // Trata erros enviados pelo servidor
    socket.on('error', (data) => showNotification(data.message, 'error'));
});
