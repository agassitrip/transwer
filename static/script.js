document.addEventListener('DOMContentLoaded', () => {
    // Establish connection with the backend
    const socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port);

    // --- STATE MANAGEMENT ---
    let currentConfig = {};
    let isListening = false;

    // --- UI ELEMENTS ---
    const settingsBtn = document.getElementById('settings-btn');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const cancelSettingsBtn = document.getElementById('cancel-settings-btn');
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const settingsModal = document.getElementById('settings-modal');
    
    // Settings fields
    const apiKeyInput = document.getElementById('api-key');
    const audioDeviceSelect = document.getElementById('audio-device-select');
    const numSuggestionsSelect = document.getElementById('num-suggestions');
    const answerDelayInput = document.getElementById('answer-delay');
    const languageSelect = document.getElementById('language-select');
    const disableTranslationCb = document.getElementById('disable-translation-cb');

    // Display containers
    const sttContainer = document.getElementById('stt-container');
    const translationContainer = document.getElementById('translation-container');
    const suggestionsContainer = document.getElementById('suggestions-container');
    const translationWrapper = document.getElementById('translation-wrapper');
    const suggestionsWrapper = document.getElementById('suggestions-wrapper');

    // Indicators & Notifications
    const apiStatusIndicator = document.getElementById('api-status-indicator');
    const apiStatusText = document.getElementById('api-status-text');
    const notification = document.getElementById('notification');

    // --- ICONS ---
    const playIcon = `<svg class="w-12 h-12" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"></path></svg>`;
    const pauseIcon = `<svg class="w-12 h-12" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clip-rule="evenodd"></path></svg>`;

    // --- INITIALIZATION ---
    playPauseBtn.innerHTML = playIcon;

    // --- MODAL LOGIC ---
    const openModal = () => {
        settingsModal.classList.remove('hidden');
        settingsModal.classList.add('flex');
    };
    const closeModal = () => {
        settingsModal.classList.add('hidden');
        settingsModal.classList.remove('flex');
    };
    
    settingsBtn.addEventListener('click', openModal);
    closeModalBtn.addEventListener('click', closeModal);
    cancelSettingsBtn.addEventListener('click', closeModal);
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) closeModal();
    });

    // --- SETTINGS & CONFIG LOGIC ---
    function updateUIFromConfig(config) {
        apiKeyInput.value = config.apiKey || '';
        audioDeviceSelect.value = config.audio_device_id || '';
        numSuggestionsSelect.value = config.num_suggestions || 3;
        answerDelayInput.value = config.answer_delay_ms || 500;
        languageSelect.value = config.target_language || 'en';
        disableTranslationCb.checked = config.disable_translation || false;
        
        // Apply visibility changes
        const isDisabled = disableTranslationCb.checked;
        translationWrapper.classList.toggle('hidden', isDisabled);
        suggestionsWrapper.classList.toggle('hidden', isDisabled);

        renderSuggestionBoxes(config.num_suggestions);
    }

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

    disableTranslationCb.addEventListener('change', (e) => {
        const isDisabled = e.target.checked;
        translationWrapper.classList.toggle('hidden', isDisabled);
        suggestionsWrapper.classList.toggle('hidden', isDisabled);
        // Also save this specific change immediately for persistence
        socket.emit('save_config', { disable_translation: isDisabled });
    });

    // --- CORE LISTENING LOGIC ---
    playPauseBtn.addEventListener('click', () => {
        if (isListening) {
            socket.emit('stop_listening');
        } else {
            // Send the current client-side config to the server when starting
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

    // --- UI UPDATE FUNCTIONS ---
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

    function updateAndScroll(element, text) {
        element.textContent = text;
        element.scrollTop = element.scrollHeight;
    }

    function updateApiStatus(status, message) {
        apiStatusIndicator.className = 'w-3 h-3 rounded-full transition-colors duration-500'; // Reset
        apiStatusIndicator.classList.add(`status-${status}`);
        apiStatusIndicator.title = `API Status: ${message}`;
        apiStatusText.textContent = message;
    }

    function renderSuggestionBoxes(count) {
        suggestionsContainer.innerHTML = '';
        for (let i = 0; i < count; i++) {
            const box = document.createElement('div');
            box.className = 'suggestion-box';
            suggestionsContainer.appendChild(box);
        }
    }

    function updateSuggestions(suggestions = []) {
        const boxes = suggestionsContainer.querySelectorAll('.suggestion-box');
        boxes.forEach((box, i) => {
            box.textContent = suggestions[i] || '';
        });
    }

    // --- SOCKET.IO EVENT HANDLERS ---
    socket.on('connect', () => console.log('Connected to server!'));

    socket.on('config_update', (config) => {
        console.log('Received config from server:', config);
        currentConfig = config;
        updateUIFromConfig(config);
    });
    
    socket.on('audio_devices_list', (devices) => {
        console.log('Received audio devices:', devices);
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
        // Set the selected value based on current config
        if (currentConfig.audio_device_id) {
            audioDeviceSelect.value = currentConfig.audio_device_id;
        }
    });

    socket.on('listening_started', () => {
        isListening = true;
        playPauseBtn.innerHTML = pauseIcon;
        playPauseBtn.classList.add('listening');
    });

    socket.on('listening_stopped', () => {
        isListening = false;
        playPauseBtn.innerHTML = playIcon;
        playPauseBtn.classList.remove('listening');
    });
    
    socket.on('notification', (data) => showNotification(data.message, data.status));
    
    socket.on('api_status', (data) => updateApiStatus(data.status, data.message));

    socket.on('update_stt', (data) => updateAndScroll(sttContainer, data.text));
    socket.on('update_translation', (data) => updateAndScroll(translationContainer, data.text));
    socket.on('update_suggestions', (data) => updateSuggestions(data.suggestions));
    
    socket.on('error', (data) => showNotification(data.message, 'error'));
});
