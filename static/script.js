// script.js
let isTabLocked = true;
let lockDuration = 30; // seconds
let tabLockEndTime = Date.now() + lockDuration * 1000;
let tabSwitchCount = 0;
const maxTabWarnings = 3;
let mediaRecorder;
let recordedChunks = [];
let tabHiddenTime = null;
let isAlertShowing = false;
let isWarningOpen = false;


// 🛑 Disable Right Click
document.addEventListener("contextmenu", (e) => e.preventDefault());

// 🛑 Block Dev Tools & Shortcut Keys
document.addEventListener("keydown", (e) => {
    // F12
    if (e.key === "F12") {
        e.preventDefault();
    }
    // Ctrl+Shift+I/J/U or Ctrl+U
    if ((e.ctrlKey && e.shiftKey && (e.key === "I" || e.key === "J" || e.key === "C")) || (e.ctrlKey && e.key === "U")) {
        e.preventDefault();
    }
});

// ✅ Unlock After Duration
function unlockTabAfterDuration() {
    setTimeout(() => {
        isTabLocked = false;
        const info = document.getElementById("tab-lock-info");
        if (info) info.textContent = "Tab is now unlocked.";
        console.log("Tab is now unlocked.");
    }, lockDuration * 1000);
}

// 🕒 Show Countdown Timer
function updateTabLockTimer() {
    const timerSpan = document.getElementById("tab-timer");
    if (!timerSpan) return;

    const interval = setInterval(() => {
        const remaining = Math.max(0, Math.ceil((tabLockEndTime - Date.now()) / 1000));
        timerSpan.textContent = remaining;

        if (remaining <= 0) {
            clearInterval(interval);
            document.getElementById("tab-lock-info").textContent = "Tab is now unlocked.";
        }
    }, 1000);
}

document.addEventListener("DOMContentLoaded", function () {
    // Your existing init calls
    updateStatus();
    updateObjectMetrics();
    updateAudioMetrics();
    getObjectDetections();

    setInterval(updateStatus, 3000);
    setInterval(updateObjectMetrics, 5000);
    setInterval(updateAudioMetrics, 5000);
    setInterval(getObjectDetections, 5000);

    // Lock + timer logic
    unlockTabAfterDuration();
    updateTabLockTimer();
});

document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabs = document.querySelectorAll('.tab');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(t => {
                t.classList.remove('active');
            });
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Hide all tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Show the corresponding tab content
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Recalibration button
    const calibrateBtn = document.getElementById('calibrate-btn');
    if (calibrateBtn) {
        calibrateBtn.addEventListener('click', recalibrate);
    }
    
    // Audio monitoring buttons
    const checkHeadphonesBtn = document.getElementById('check-headphones-btn');
    const testMicBtn = document.getElementById('test-mic-btn');
    const startAudioBtn = document.getElementById('start-audio-btn');
    const stopAudioBtn = document.getElementById('stop-audio-btn');
    const debugAudioLevelsBtn = document.getElementById('debug-audio-levels-btn');
    const saveSpokenContentBtn = document.getElementById('save-spoken-content-btn');

    if (checkHeadphonesBtn) {
        checkHeadphonesBtn.addEventListener('click', checkHeadphones);
    }

    if (testMicBtn) {
        testMicBtn.addEventListener('click', testMicrophone);
    }

    if (startAudioBtn) {
        startAudioBtn.addEventListener('click', startAudioMonitoring);
    }

    if (stopAudioBtn) {
        stopAudioBtn.addEventListener('click', stopAudioMonitoring);
    }
    
    if (debugAudioLevelsBtn) {
        debugAudioLevelsBtn.addEventListener('click', debugAudioLevels);
    }
    
    if (saveSpokenContentBtn) {
        saveSpokenContentBtn.addEventListener('click', saveSpokenContent);
    }
    
    // Object detection toggle button
    const toggleObjectDetectionBtn = document.getElementById('toggle-object-detection-btn');
    if (toggleObjectDetectionBtn) {
        toggleObjectDetectionBtn.addEventListener('click', toggleObjectDetection);
    }
    
    // Save object detections button
    const saveObjectDetectionsBtn = document.getElementById('save-object-detections-btn');
    if (saveObjectDetectionsBtn) {
        saveObjectDetectionsBtn.addEventListener('click', saveObjectDetections);
    }
    
    // Get object detections button
    const getObjectDetectionsBtn = document.getElementById('get-object-detections-btn');
    if (getObjectDetectionsBtn) {
        getObjectDetectionsBtn.addEventListener('click', getObjectDetections);
    }
    
    // User management buttons
    const deleteUserBtns = document.querySelectorAll('.delete-user-btn');
    if (deleteUserBtns) {
        deleteUserBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const userId = this.getAttribute('data-user-id');
                if (confirm(`Are you sure you want to delete user #${userId}?`)) {
                    window.location.href = `/delete_user/${userId}`;
                }
            });
        });
    }
    
    // Auto-calibrate on page load after a short delay
    setTimeout(recalibrate, 1000);
    
    // Update status every 1.5 seconds
    setInterval(updateStatus, 1500);
    
    // Update object detection metrics every 2 seconds
    setInterval(updateObjectMetrics, 2000);
    
    // Update audio metrics every 3 seconds
    setInterval(updateAudioMetrics, 3000);
    
    // Initial status update
    updateStatus();
    
    // Auto-hide flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.flash-message');
    if (flashMessages.length > 0) {
        setTimeout(() => {
            flashMessages.forEach(msg => {
                msg.style.transition = 'opacity 1s';
                msg.style.opacity = '0';
                setTimeout(() => {
                    msg.style.display = 'none';
                }, 1000);
            });
        }, 5000);
    }

    // Initialize tab switching
    initializeTabSwitching();
});

// Function to start verification for a specific user
function startVerification(userId) {
    fetch(`/start_verification/${userId}`)
        .then(response => {
            if (response.ok) {
                // Add a verification status message to the UI
                const statusArea = document.getElementById('verification-status');
                if (statusArea) {
                    statusArea.innerText = `User #${userId}: Verification in progress...`;
                    statusArea.className = 'status-indicator verifying';
                }
            } else {
                alert('Failed to start verification.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while trying to start verification.');
        });
}

// Function to trigger recalibration
function recalibrate() {
    fetch('/calibrate')
        .then(response => {
            if (response.ok) {
                document.getElementById('eye-status').innerText = 'Eyes: Calibrating... Please look at the center of the screen.';
                document.getElementById('eye-status').className = 'status-indicator status-box no-face';
                document.getElementById('head-status').innerText = 'Head: Calibrating... Please look straight ahead.';
                document.getElementById('head-status').className = 'status-indicator status-box no-face';
            } else {
                alert('Failed to start calibration.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while trying to recalibrate.');
        });
}

// Function to update status information from the server
function updateStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            // ==== EYE STATUS ====
            const eyeStatusElement = document.getElementById('eye-status');
            if (eyeStatusElement) {
                eyeStatusElement.innerText = 'Eyes: ' + (data.gaze_status || 'Unknown Status');
                eyeStatusElement.className = 'status-indicator status-box ' + (data.status_class || 'no-face');
            }

            // Detailed eye tracking metrics
            if (document.getElementById('current-gaze')) {
                document.getElementById('current-gaze').innerText = data.gaze_status || 'Unknown';
            }

            if (document.getElementById('detailed-eye-status')) {
                document.getElementById('detailed-eye-status').innerText = data.gaze_status || 'Not detected';
                document.getElementById('detailed-eye-status').className = 'status-box ' + (data.status_class || 'no-face');
            }

            // Extract confidence from gaze status (e.g., "Looking at Screen (85% confidence)")
            const confidenceMatch = (data.gaze_status || '').match(/\((\d+)%/);
            const gazeConfidence = confidenceMatch ? parseInt(confidenceMatch[1]) : 0;
            
            if (document.getElementById('gaze-confidence-progress')) {
                document.getElementById('gaze-confidence-progress').style.width = gazeConfidence + '%';
            }
            
            if (document.getElementById('gaze-confidence-text')) {
                document.getElementById('gaze-confidence-text').innerText = gazeConfidence + '%';
            }

            if (document.getElementById('blink-rate')) {
                document.getElementById('blink-rate').innerText = 
                    (data.blink_rate || 0).toFixed(1) + ' blinks/min';
            }

            // ==== HEAD STATUS ====
            const headStatusElement = document.getElementById('head-status');
            if (headStatusElement) {
                headStatusElement.innerText = 'Head: ' + (data.head_status || 'Unknown Status');
                const headClass = data.head_direction === 'head_straight' ? 'head-straight' : 'head-moved';
                headStatusElement.className = 'status-indicator status-box ' + headClass;
            }

            // Update head position
            if (document.getElementById('head-position')) {
                document.getElementById('head-position').innerText = 
                    formatHeadDirection(data.head_direction || 'unknown');
            }

            // Update head movement count
            if (document.getElementById('head-movement-count')) {
                document.getElementById('head-movement-count').innerText = 
                    data.head_movement_count || 0;
            }

            if (document.getElementById('total-head-movements')) {
                document.getElementById('total-head-movements').innerText = 
                    data.head_movement_count || 0;
            }

            // Update head straight time percentage
            const straightPercentage = data.straight_time_percentage || 0;
            if (document.getElementById('head-straight-progress')) {
                document.getElementById('head-straight-progress').style.width = straightPercentage + '%';
            }
            
            if (document.getElementById('head-straight-text')) {
                document.getElementById('head-straight-text').innerText = straightPercentage.toFixed(1) + '%';
            }

            // Update head movement status
            if (document.getElementById('head-movement-status')) {
                const status = data.head_direction === 'head_straight' ? 'Stable' : 'Moving';
                document.getElementById('head-movement-status').innerText = status;
                document.getElementById('head-movement-status').className = 
                    'status-box ' + (data.head_direction === 'head_straight' ? 'head-straight' : 'head-moved');
            }

            // Update head direction in detailed view
            if (document.getElementById('head-direction')) {
                document.getElementById('head-direction').innerText =
                    formatHeadDirection(data.head_direction || 'unknown');
            }

            // Update movement intensity (based on how far from straight position)
            const movementIntensity = data.head_direction === 'head_straight' ? 0 : 
                Math.min(100, (1 - (data.straight_time_percentage || 0) / 100) * 200);
            
            if (document.getElementById('head-movement-intensity-progress')) {
                document.getElementById('head-movement-intensity-progress').style.width = movementIntensity + '%';
            }
            
            if (document.getElementById('head-movement-intensity-text')) {
                document.getElementById('head-movement-intensity-text').innerText = movementIntensity.toFixed(1) + '%';
            }

            // Update head position history
            if (document.getElementById('head-position-history')) {
                const historyElement = document.getElementById('head-position-history');
                if (data.head_direction && data.head_direction !== 'unknown') {
                    const timestamp = new Date().toLocaleTimeString();
                    const newEntry = document.createElement('div');
                    newEntry.className = 'history-entry';
                    newEntry.innerHTML = `${timestamp}: ${formatHeadDirection(data.head_direction)}`;
                    
                    // Keep only last 5 entries
                    while (historyElement.childNodes.length >= 5) {
                        historyElement.removeChild(historyElement.firstChild);
                    }
                    historyElement.appendChild(newEntry);
                } else if (historyElement.innerText === 'No movements recorded') {
                    // Keep the default text if no movements
                    historyElement.innerText = 'No movements recorded';
                }
            }

            // ==== EYE TRACKING METRICS ====
            if (document.getElementById('total-reading-time')) {
                const totalReadingTime = parseFloat(data.total_reading_time || 0);
                document.getElementById('total-reading-time').innerText =
                    totalReadingTime.toFixed(1) + ' seconds';
            }
            
            if (document.getElementById('reading-sessions')) {
                document.getElementById('reading-sessions').innerText =
                    parseInt(data.reading_sessions || 0);
            }

            const continuousReading = parseFloat(data.continuous_reading_time || 0);
            const continuousProgress = Math.min((continuousReading / 30) * 100, 100);
            if (document.getElementById('continuous-reading-progress')) {
                document.getElementById('continuous-reading-progress').style.width = continuousProgress + '%';
            }
            
            if (document.getElementById('continuous-reading-text')) {
                document.getElementById('continuous-reading-text').innerText = continuousReading.toFixed(1) + 's';
            }

            const onScreenPercentage = parseFloat(data.on_screen_percentage || 0);
            if (document.getElementById('on-screen-progress')) {
                document.getElementById('on-screen-progress').style.width = onScreenPercentage + '%';
            }
            
            if (document.getElementById('on-screen-text')) {
                document.getElementById('on-screen-text').innerText = onScreenPercentage.toFixed(1) + '%';
            }

            // ==== FACE VERIFICATION ====
            const verificationStatus = document.getElementById('verification-status');
            const faceVerificationStatus = document.getElementById('face-verification-status');

            if (data.verification) {
                const verification = data.verification;

                if (verification.verification_mode && !verification.verification_complete) {
                    if (verificationStatus) {
                        verificationStatus.innerText = `User #${verification.user_id}: Verifying... ${verification.verification_time_remaining.toFixed(1)}s remaining`;
                        verificationStatus.className = 'status-indicator verifying';
                    }

                    if (faceVerificationStatus) {
                        faceVerificationStatus.innerText = 'Verification in progress';
                    }

                    if (document.getElementById('verification-time-progress') && document.getElementById('verification-time-text')) {
                        const timePercentage = (1 - verification.verification_time_remaining / 20) * 100;
                        document.getElementById('verification-time-progress').style.width = `${timePercentage}%`;
                        document.getElementById('verification-time-text').innerText =
                            `${verification.verification_time_remaining.toFixed(1)}s remaining`;
                    }

                } else if (verification.is_verified) {
                    if (faceVerificationStatus) {
                        faceVerificationStatus.innerText = 'Verified';
                    }
                    
                    if (verificationStatus) {
                        verificationStatus.innerText = `User #${verification.user_id}: Verified`;
                        verificationStatus.className = 'status-indicator verified';
                    }
                } else {
                    if (faceVerificationStatus) {
                        faceVerificationStatus.innerText = verification.verification_status || 'Not Verified';
                    }
                    
                    if (verificationStatus) {
                        verificationStatus.innerText = `User #${verification.user_id}: Not Verified`;
                        verificationStatus.className = 'status-indicator not-verified';
                    }
                }

                const confidencePercentage = verification.verification_confidence * 100 || 0;
                if (document.getElementById('verification-confidence-progress')) {
                    document.getElementById('verification-confidence-progress').style.width = confidencePercentage + '%';
                }
                
                if (document.getElementById('verification-confidence-text')) {
                    document.getElementById('verification-confidence-text').innerText = confidencePercentage.toFixed(1) + '%';
                }

                if (document.getElementById('max-verification-confidence')) {
                    const maxConfidencePercentage = verification.max_confidence * 100 || 0;
                    document.getElementById('max-verification-confidence').innerText = maxConfidencePercentage.toFixed(1) + '%';
                }

                if (document.getElementById('verification-checks')) {
                    document.getElementById('verification-checks').innerText = verification.verification_checks || 0;
                }

                if (document.getElementById('last-verification-time')) {
                    if (verification.last_verification_time) {
                        const verificationTime = new Date(verification.last_verification_time * 1000);
                        document.getElementById('last-verification-time').innerText = verificationTime.toLocaleTimeString();
                    } else {
                        document.getElementById('last-verification-time').innerText = 'Never';
                    }
                }
            }

            // ==== AUDIO ====
            if (data.audio) {
                const audioStatusElement = document.getElementById('audio-status');
                if (audioStatusElement) {
                    if (data.audio.audio_monitoring_active) {
                        audioStatusElement.innerText = 'Audio: Monitoring';
                        audioStatusElement.className = 'status-indicator status-box active';
                    } else {
                        audioStatusElement.innerText = 'Audio: Inactive';
                        audioStatusElement.className = 'status-indicator status-box inactive';
                    }
                }

                const monitoringStatusElement = document.getElementById('audio-monitoring-status');
                if (monitoringStatusElement) {
                    monitoringStatusElement.innerText = data.audio.audio_monitoring_active
                        ? 'Active - Monitoring Speech'
                        : 'Inactive';
                    monitoringStatusElement.className = data.audio.audio_monitoring_active
                        ? 'status-indicator verified'
                        : 'status-indicator not-verified';
                }

                if (document.getElementById('words-detected')) {
                    document.getElementById('words-detected').innerText = data.audio.total_words_detected || 0;
                }
                
                if (document.getElementById('sentences-detected')) {
                    document.getElementById('sentences-detected').innerText = data.audio.total_sentences || 0;
                }

                const lastSpokenElement = document.getElementById('last-spoken-text');
                if (lastSpokenElement) {
                    lastSpokenElement.innerText = data.audio.last_spoken_text || 'None';
                    if (data.audio.last_detection_time &&
                        (Date.now() / 1000 - data.audio.last_detection_time) < 5) {
                        lastSpokenElement.className = 'speech-text recent';
                        setTimeout(() => {
                            lastSpokenElement.className = 'speech-text';
                        }, 3000);
                    }
                }

                const commonWordsElement = document.getElementById('common-words');
                if (commonWordsElement && data.audio.most_common_words) {
                    if (data.audio.most_common_words.length > 0) {
                        let wordsList = '';
                        data.audio.most_common_words.forEach(wordData => {
                            wordsList += `<span class="word-item">${wordData[0]}: ${wordData[1]}</span>`;
                        });
                        commonWordsElement.innerHTML = wordsList;
                    } else {
                        commonWordsElement.innerText = 'None';
                    }
                }

                if (document.getElementById('noise-level-progress') && document.getElementById('noise-level-text')) {
                    const noisePercentage = Math.min((data.audio.background_noise_level / 1000) * 100, 100);
                    document.getElementById('noise-level-progress').style.width = noisePercentage + '%';
                    document.getElementById('noise-level-text').innerText = data.audio.background_noise_level;
                }

                const headphonesStatusElement = document.getElementById('headphones-status');
                if (headphonesStatusElement) {
                    if (data.audio.headphones_with_mic) {
                        headphonesStatusElement.innerText = 'Headphones with Microphone Detected';
                        headphonesStatusElement.className = 'status-indicator verified';
                    } else if (data.audio.headphones_detected) {
                        headphonesStatusElement.innerText = 'Headphones Detected (No Microphone)';
                        headphonesStatusElement.className = 'status-indicator warning';
                    } else {
                        headphonesStatusElement.innerText = 'No Headphones Detected';
                        headphonesStatusElement.className = 'status-indicator not-verified';
                    }
                }
                
                if (document.getElementById('time-since-detection')) {
                    if (data.audio.last_detection_time) {
                        document.getElementById('time-since-detection').innerText = 
                            formatTimeSince(data.audio.last_detection_time);
                    } else {
                        document.getElementById('time-since-detection').innerText = 'Never';
                    }
                }
            }

            // ==== OBJECT DETECTION ====
            if (data.object_detection) {
                const objectDetectionStatus = document.getElementById('object-detection-status');
                if (objectDetectionStatus) {
                    objectDetectionStatus.innerText = data.object_detection.object_detection_active ? 'Active' : 'Inactive';
                    objectDetectionStatus.className = data.object_detection.object_detection_active 
                        ? 'status-indicator verified' 
                        : 'status-indicator not-verified';
                }
                
                if (document.getElementById('objects-detected')) {
                    document.getElementById('objects-detected').innerText = data.object_detection.objects_detected || 0;
                }
                
                if (document.getElementById('last-object-detection')) {
                    if (data.object_detection.last_object_detection) {
                        document.getElementById('last-object-detection').innerText = 
                            formatTimeSince(data.object_detection.last_object_detection);
                    } else {
                        document.getElementById('last-object-detection').innerText = 'Never';
                    }
                }
                
                // Display recent objects detected
                const recentObjectsElement = document.getElementById('recent-objects');
                if (recentObjectsElement && data.object_detection.recent_objects) {
                    if (data.object_detection.recent_objects.length > 0) {
                        let objectsList = '';
                        data.object_detection.recent_objects.slice(0, 5).forEach(objData => {
                            objectsList += `<div class="object-item">
                                <span class="object-name">${objData.object}</span>
                                <span class="object-time">${formatTimeSince(objData.timestamp)}</span>
                                <span class="object-confidence">${(objData.confidence * 100).toFixed(1)}%</span>
                            </div>`;
                        });
                        recentObjectsElement.innerHTML = objectsList;
                    } else {
                        recentObjectsElement.innerText = 'None';
                    }
                }
                
                // Display most common objects
                const commonObjectsElement = document.getElementById('common-objects');
                if (commonObjectsElement && data.object_detection.most_common_objects) {
                    if (data.object_detection.most_common_objects.length > 0) {
                        let objectsList = '';
                        data.object_detection.most_common_objects.forEach(objData => {
                            objectsList += `<span class="object-item">${objData[0]}: ${objData[1]}</span>`;
                        });
                        commonObjectsElement.innerHTML = objectsList;
                    } else {
                        commonObjectsElement.innerText = 'None';
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error fetching status:', error);
        });
}

// Function to update object detection metrics
function updateObjectMetrics() {
    fetch('/object_metrics')
        .then(response => response.json())
        .then(data => {
            // Update the object detection metrics in the UI
            if (document.getElementById('objects-detected')) {
                document.getElementById('objects-detected').innerText = data.total_objects_detected || 0;
            }
            
            if (document.getElementById('object-detection-status')) {
                const statusElement = document.getElementById('object-detection-status');
                statusElement.innerText = data.detection_active ? 'Active' : 'Inactive';
                statusElement.className = data.detection_active ? 'status-indicator verified' : 'status-indicator not-verified';
            }
            
            if (document.getElementById('last-object-detection')) {
                if (data.time_since_last_detection) {
                    document.getElementById('last-object-detection').innerText = 
                        formatTimeSince(Date.now()/1000 - data.time_since_last_detection);
                } else {
                    document.getElementById('last-object-detection').innerText = 'Never';
                }
            }
            
            if (document.getElementById('object-classes-detected')) {
                const classesElement = document.getElementById('object-classes-detected');
                if (data.detected_classes && data.detected_classes.length > 0) {
                    let classesList = '';
                    data.detected_classes.forEach(classData => {
                        classesList += `<span class="class-item">${classData[0]}: ${classData[1]}</span>`;
                    });
                    classesElement.innerHTML = classesList;
                } else {
                    classesElement.innerText = 'None';
                }
            }
        })
        .catch(error => {
            console.error('Error fetching object metrics:', error);
        });
}

// Function to update audio metrics
function updateAudioMetrics() {
    fetch('/audio_metrics')
        .then(response => response.json())
        .then(data => {
            // Update audio monitoring status
            const audioMonitoringStatus = document.getElementById('audio-monitoring-status');
            if (audioMonitoringStatus) {
                audioMonitoringStatus.innerText = data.audio_monitoring_active ? 'Active - Monitoring Speech' : 'Inactive';
                audioMonitoringStatus.className = data.audio_monitoring_active ? 'status-indicator verified' : 'status-indicator not-verified';
            }
            
            // Update word count
            if (document.getElementById('words-detected')) {
                document.getElementById('words-detected').innerText = data.total_words_detected || 0;
            }
            
            // Update sentence count
            if (document.getElementById('sentences-detected')) {
                document.getElementById('sentences-detected').innerText = data.total_sentences || 0;
            }
            
            // Update last spoken text
            const lastSpokenElement = document.getElementById('last-spoken-text');
            if (lastSpokenElement) {
                lastSpokenElement.innerText = data.last_spoken_text || 'None';
                
                // Highlight recent text
                if (data.last_detection_time && (Date.now() / 1000 - data.last_detection_time) < 5) {
                    lastSpokenElement.className = 'speech-text recent';
                    setTimeout(() => {
                        lastSpokenElement.className = 'speech-text';
                    }, 3000);
                }
            }
            
            // Update most common words
            const commonWordsElement = document.getElementById('common-words');
            if (commonWordsElement && data.most_common_words) {
                if (data.most_common_words.length > 0) {
                    let wordsList = '';
                    data.most_common_words.forEach(wordData => {
                        wordsList += `<span class="word-item">${wordData[0]}: ${wordData[1]}</span>`;
                    });
                    commonWordsElement.innerHTML = wordsList;
                } else {
                    commonWordsElement.innerText = 'None';
                }
            }
            
            // Update background noise level
            if (document.getElementById('noise-level-progress') && document.getElementById('noise-level-text')) {
                const noisePercentage = Math.min((data.background_noise_level / 1000) * 100, 100);
                document.getElementById('noise-level-progress').style.width = noisePercentage + '%';
                document.getElementById('noise-level-text').innerText = data.background_noise_level;
            }
            
            // Update time since last detection
            if (document.getElementById('time-since-detection')) {
                if (data.last_detection_time) {
                    document.getElementById('time-since-detection').innerText = formatTimeSince(data.last_detection_time);
                } else {
                    document.getElementById('time-since-detection').innerText = 'Never';
                }
            }
        })
        .catch(error => {
            console.error('Error fetching audio metrics:', error);
        });
}

function formatTimeSince(timestamp) {
    if (!timestamp) return 'Unknown time';
    
    const seconds = Date.now() / 1000 - timestamp;
    
    if (seconds < 60) {
        return `${seconds.toFixed(1)} seconds ago`;
    } else if (seconds < 3600) {
        return `${(seconds / 60).toFixed(1)} minutes ago`;
    } else if (seconds < 86400) {
        return `${(seconds / 3600).toFixed(1)} hours ago`;
    } else {
        return `${(seconds / 86400).toFixed(1)} days ago`;
    }
}

// Format head direction for display
function formatHeadDirection(direction) {
    if (!direction) return 'Unknown';
    
    // Convert from code to readable format
    switch(direction) {
        case 'head_straight': return 'Straight';
        case 'looking_up': return 'Looking Up';
        case 'looking_down': return 'Looking Down';
        case 'looking_left': return 'Looking Left';
        case 'looking_right': return 'Looking Right';
        case 'head_tilted_left': return 'Tilted Left';
        case 'head_tilted_right': return 'Tilted Right';
        case 'no_face': return 'No Face Detected';
        default: return direction.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
}

// Function to check headphones status
function checkHeadphones() {
    fetch('/check_headphones')
        .then(response => response.json())
        .then(data => {
            const headphonesStatusElement = document.getElementById('headphones-status');
            
            if (data.headphones_with_mic) {
                headphonesStatusElement.innerText = 'Headphones with Microphone Detected';
                headphonesStatusElement.className = 'status-indicator verified';
            } else if (data.headphones_detected) {
                headphonesStatusElement.innerText = 'Headphones Detected (No Microphone)';
                headphonesStatusElement.className = 'status-indicator warning';
            } else {
                headphonesStatusElement.innerText = 'No Headphones Detected';
                headphonesStatusElement.className = 'status-indicator not-verified';
            }
        })
        .catch(error => {
            console.error('Error checking headphones:', error);
        });
}

// Function to test microphone
function testMicrophone() {
    const headphonesStatusElement = document.getElementById('headphones-status');
    headphonesStatusElement.innerText = 'Testing Microphone...';
    headphonesStatusElement.className = 'status-indicator verifying';
    
    fetch('/test_microphone')
        .then(response => response.json())
        .then(data => {
            if (data.microphone_working) {
                headphonesStatusElement.innerText = `Microphone Working (Level: ${data.audio_level.toFixed(2)})`;
                headphonesStatusElement.className = 'status-indicator verified';
            } else {
                headphonesStatusElement.innerText = 'Microphone Not Working or Not Detected';
                headphonesStatusElement.className = 'status-indicator not-verified';
            }
        })
        .catch(error => {
            console.error('Error testing microphone:', error);
            headphonesStatusElement.innerText = 'Error Testing Microphone';
            headphonesStatusElement.className = 'status-indicator not-verified';
        });
}

// Function to start audio monitoring
function startAudioMonitoring() {
    fetch('/start_audio_monitoring')
        .then(response => {
            if (response.ok) {
                const audioStatusElement = document.getElementById('audio-monitoring-status');
                audioStatusElement.innerText = 'Active - Monitoring Speech';
                audioStatusElement.className = 'status-indicator verified';
                
                // Update the audio status in the status container
                const audioStatus = document.getElementById('audio-status');
                if (audioStatus) {
                    audioStatus.innerText = 'Audio: Monitoring';
                    audioStatus.className = 'status-indicator status-box active';
                }
            } else {
                alert('Failed to start audio monitoring.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while trying to start audio monitoring.');
        });
}

// Function to stop audio monitoring
function stopAudioMonitoring() {
    fetch('/stop_audio_monitoring')
        .then(response => {
            if (response.ok) {
                const audioStatusElement = document.getElementById('audio-monitoring-status');
                audioStatusElement.innerText = 'Inactive';
                audioStatusElement.className = 'status-indicator not-verified';
                
                // Update the audio status in the status container
                const audioStatus = document.getElementById('audio-status');
                if (audioStatus) {
                    audioStatus.innerText = 'Audio: Inactive';
                    audioStatus.className = 'status-indicator status-box inactive';
                }
            } else {
                alert('Failed to stop audio monitoring.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while trying to stop audio monitoring.');
        });
}

// Function to toggle object detection
function toggleObjectDetection() {
    fetch('/toggle_object_detection')
        .then(response => {
            if (response.ok) {
                // Refresh object metrics immediately
                updateObjectMetrics();
            } else {
                alert('Failed to toggle object detection.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while trying to toggle object detection.');
        });
}

// Function to save object detections
function saveObjectDetections() {
    fetch('/save_object_detections')
        .then(response => {
            if (response.ok) {
                alert('Object detections saved successfully.');
            } else {
                alert('Failed to save object detections.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while trying to save object detections.');
        });
}

// Function to get object detections
function getObjectDetections() {
    fetch('/object_detections')
        .then(response => response.json())
        .then(data => {
            console.log("Object detection data:", data); // ADD THIS LINE

            const container = document.getElementById('object-detections-list');
            if (container) {
                container.innerHTML = ''; // Clear previous data
                if (data.length > 0) {
                    let detectionsHTML = '';
                    data.slice(-20).reverse().forEach(detection => {
                        const detectionTime = new Date(detection.timestamp * 1000).toLocaleTimeString();
                        detectionsHTML += `
                            <div class="detection-item">
                                <strong>${detection.object}</strong> – 
                                <span>${(detection.confidence * 100).toFixed(1)}%</span> @ 
                                <span>${detectionTime}</span>
                            </div>`;
                    });
                    container.innerHTML = detectionsHTML;
                } else {
                    container.innerText = 'No detections found.';
                }
            }
        })
        .catch(error => {
            console.error('Error fetching object detections:', error);
            const container = document.getElementById('object-detections-list');
            if (container) {
                container.innerText = 'Error loading detections.';
            }
        });
}

document.addEventListener("DOMContentLoaded", function () {
    // Other functions like updateStatus(), updateObjectMetrics(), etc.

    getObjectDetections(); // ✅ Call once on load

    // ✅ Call every 5 seconds to update list
    setInterval(getObjectDetections, 5000);
});

window.addEventListener("blur", () => {
    // User switched or minimized tab
    tabHiddenTime = Date.now();
});
window.addEventListener("focus", () => {
    if (tabHiddenTime && !isWarningOpen) {
        tabHiddenTime = null;
        tabSwitchCount++;
        console.log("Tab switch count:", tabSwitchCount);

        if (tabSwitchCount < maxTabWarnings) {
            showTabWarning(`⚠️ Warning ${tabSwitchCount}/2: Please do not switch tabs!`);
        } else {
            showTabWarning("❌ You switched tabs 3 times. The exam is now terminated.", true);
        }
    }
});

function endExam() {
    // Stop recording FIRST to trigger download
    stopRecording();

    // Delay cleanup slightly to allow onstop to finish setting the download button
    setTimeout(() => {
        // Stop all intervals
        let highest = setInterval(() => {}, 1000);
        for (let i = 0; i <= highest; i++) clearInterval(i);

        document.documentElement.innerHTML = `
            <body style="text-align:center; padding:50px;">
                <h1 style="color:red;">❌ Exam Ended</h1>
                <p>You switched tabs too many times. The exam has been terminated.</p>
                <a id="download-recording-btn" style="display:inline-block; margin-top:20px; font-size:18px;"
                   class="btn btn-primary" download="exam_recording.webm">Download Recording</a>
            </body>
        `;

        // Re-assign blob if it already exists
        const blob = new Blob(recordedChunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const downloadBtn = document.getElementById("download-recording-btn");
        if (downloadBtn) {
            downloadBtn.href = url;
        }
    }, 500); // Small delay to ensure recording finishes
}


function startRecording() {
    const videoElement = document.getElementById("video");

    // Create canvas to draw frames
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.width || 640;
    canvas.height = videoElement.height || 480;
    const ctx = canvas.getContext('2d');

    const stream = canvas.captureStream(15); // 15 fps
    mediaRecorder = new MediaRecorder(stream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = function (e) {
        if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = function () {
        const blob = new Blob(recordedChunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const downloadBtn = document.getElementById("download-recording-btn");

        downloadBtn.href = url;
        downloadBtn.download = "exam_recording.webm";
        downloadBtn.style.display = "inline-block";
    };

    mediaRecorder.start();

    // 🖌️ Draw video frames to canvas
    function drawFrame() {
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        if (mediaRecorder.state !== "inactive") {
            requestAnimationFrame(drawFrame);
        }
    }

    drawFrame();
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
}

document.addEventListener("DOMContentLoaded", function () {
    // Start recording automatically when test begins
    startRecording();

    // Optional: Bind to button for manual stop
    const downloadBtn = document.getElementById("download-recording-btn");
    if (downloadBtn) {
        downloadBtn.addEventListener('click', stopRecording);
    }
});

function showTabWarning(message, shouldEndExam = false) {
    const modal = document.getElementById('tab-warning-modal');
    const text = document.getElementById('tab-warning-text');
    const closeBtn = document.getElementById('close-warning-btn');

    if (!modal || !text || !closeBtn) return;

    text.innerText = message;
    modal.style.display = 'block';
    isWarningOpen = true;

    closeBtn.onclick = () => {
        modal.style.display = 'none';
        isWarningOpen = false;

        if (shouldEndExam) {
            endExam();
        }
    };
}

// Tab switching functionality
function initializeTabSwitching() {
    const tabs = document.querySelectorAll('.tab');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            const contentId = tab.getAttribute('data-tab');
            document.getElementById(contentId).classList.add('active');
        });
    });
}
