import { useState } from "react";
import axios from "axios";
import "./App.css";

// --- COMPOSANT CHAT (DÉFINI EN DEHORS DE APP POUR FIXER LE BUG DE FOCUS) ---
const ChatComponent = ({ 
    messages, 
    input, 
    setInput, 
    category, 
    setCategory, 
    language, 
    setLanguage, 
    onSend 
}) => (
    <div className="card chat-card-full">
        <h2>Assistant Cardiologue</h2>
        
        <div className="chat-box">
            {messages.map((m, i) => (
                <div key={i} className={`chat-message ${m.role}`}>
                    <b>{m.role === "user" ? "Moi" : "IA"}:</b> {m.content}
                </div>
            ))}
        </div>
        
        <form onSubmit={onSend} className="chat-form-container">
            {/* Ligne 1 : Filtres */}
            <div className="chat-filters">
                <select 
                    value={category} 
                    onChange={(e) => setCategory(e.target.value)}
                    className="chat-select"
                    title="Catégorie médicale"
                >
                    <option value="general">Général</option>
                    <option value="arrhythmias">Arythmies</option>
                    <option value="ischemia">Ischémie</option>
                    <option value="symptoms">Symptômes</option>
                    <option value="risk_factors">Facteurs de Risque</option>
                </select>

                <select 
                    value={language} 
                    onChange={(e) => setLanguage(e.target.value)}
                    className="chat-select"
                    title="Langue de réponse"
                >
                    <option value="fr">Français</option>
                    <option value="en">English</option>
                    <option value="ar">العربية</option>
                </select>
            </div>

            {/* Ligne 2 : Saisie */}
            <div className="chat-input-row">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Posez votre question..."
                    autoFocus
                />
                <button type="submit">Envoyer</button>
            </div>
        </form>
    </div>
);

function App() {
    // --- Etats Navigation & Formulaire ---
    const [activeTab, setActiveTab] = useState("analysis"); // 'analysis' ou 'chat'
    const [step, setStep] = useState("form"); // 'form' ou 'result'
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const [form, setForm] = useState({
        first_name: "", last_name: "", age: "", sex: "",
        weight: "", height: "", symptoms: "",
    });

    const [datFile, setDatFile] = useState(null);
    const [heaFile, setHeaFile] = useState(null);

    // --- Etats Chatbot ---
    const [chatInput, setChatInput] = useState("");
    const [chatCategory, setChatCategory] = useState("general");
    const [chatLanguage, setChatLanguage] = useState("fr");
    const [chatMessages, setChatMessages] = useState([
        { role: "assistant", content: "Bonjour ! Je suis votre assistant cardiologique. Comment puis-je vous aider ?" }
    ]);

    // --- Handlers ---
    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!datFile || !heaFile) {
            alert("Veuillez uploader les fichiers .dat et .hea");
            return;
        }

        const data = new FormData();
        Object.entries(form).forEach(([k, v]) => { if (v !== "") data.append(k, v); });
        data.append("dat_file", datFile);
        data.append("hea_file", heaFile);

        try {
            setLoading(true);
            const res = await axios.post("http://127.0.0.1:8000/ecg/predict", data);
            setResult(res.data);
            setStep("result");
        } catch (err) {
            console.error(err);
            alert("Erreur serveur lors de l'analyse");
        } finally {
            setLoading(false);
        }
    };

    const handleChatSubmit = async (e) => {
        e.preventDefault();
        if (!chatInput) return;

        const userMessage = { role: "user", content: chatInput };
        setChatMessages([...chatMessages, userMessage]);
        
        const currentMsg = chatInput;
        setChatInput(""); 

        try {
            const res = await axios.post("http://127.0.0.1:8000/chatbot/api/chat", { 
                message: currentMsg, 
                language: chatLanguage,
                category: chatCategory 
            });
            
            const botMessage = { role: "assistant", content: res.data.content };
            setChatMessages((prev) => [...prev, botMessage]);
        } catch (err) {
            console.error(err);
            setChatMessages((prev) => [...prev, { role: "assistant", content: "Erreur de connexion au serveur IA." }]);
        }
    };

    return (
        <div className="page">
            {/* Barre de navigation */}
            <nav className="navbar">
                <div className="nav-title">CardioAI</div>
                <div className="nav-buttons">
                    <button 
                        className={activeTab === 'analysis' ? 'active' : ''} 
                        onClick={() => setActiveTab('analysis')}
                    >
                        Analyse ECG
                    </button>
                    <button 
                        className={activeTab === 'chat' ? 'active' : ''} 
                        onClick={() => setActiveTab('chat')}
                    >
                        Chatbot IA
                    </button>
                </div>
            </nav>

            {/* Contenu principal (Centré) */}
            <div className="content-container">
                
                {/* CAS 1 : CHAT SEUL */}
                {activeTab === "chat" && (
                    <ChatComponent 
                        messages={chatMessages}
                        input={chatInput}
                        setInput={setChatInput}
                        category={chatCategory}
                        setCategory={setChatCategory}
                        language={chatLanguage}
                        setLanguage={setChatLanguage}
                        onSend={handleChatSubmit}
                    />
                )}

                {/* CAS 2 : ANALYSE ECG */}
                {activeTab === "analysis" && (
                    <>
                        {step === "form" && (
                            <div className="card">
                                <h2 style={{textAlign: 'center', marginTop: 0}}>Nouveau Patient</h2>
                                <form onSubmit={handleSubmit} className="form">
                                    <input required name="first_name" placeholder="Prénom" onChange={handleChange} />
                                    <input required name="last_name" placeholder="Nom" onChange={handleChange} />
                                    <input required type="number" name="age" placeholder="Age" onChange={handleChange} />
                                    
                                    <div className="radio-group">
                                        <span>Sexe:</span>
                                        <label><input type="radio" name="sex" value="F" required onChange={handleChange} /> F</label>
                                        <label><input type="radio" name="sex" value="M" required onChange={handleChange} /> M</label>
                                    </div>

                                    <input required type="number" name="weight" placeholder="Poids (kg)" onChange={handleChange} />
                                    <input required type="number" name="height" placeholder="Taille (cm)" onChange={handleChange} />
                                    <textarea name="symptoms" placeholder="Symptômes (optionnel)" onChange={handleChange} />

                                    <div className="file-group">
                                        <label>Fichier .dat</label>
                                        <input type="file" accept=".dat" required onChange={(e) => setDatFile(e.target.files[0])} />
                                    </div>
                                    <div className="file-group">
                                        <label>Fichier .hea</label>
                                        <input type="file" accept=".hea" required onChange={(e) => setHeaFile(e.target.files[0])} />
                                    </div>

                                    <button type="submit" disabled={loading}>{loading ? "Analyse en cours..." : "Lancer Diagnostic"}</button>
                                </form>
                            </div>
                        )}

                        {step === "result" && (
                            <div className="result-container">
                                {/* Panneau Gauche : Résultats */}
                                <div className="left-panel card">
                                    <div className="patient-info">
                                        <h3>Résultats pour {result.patient.first_name} {result.patient.last_name}</h3>
                                        <p>Age: {result.patient.age} ans | Sexe: {result.patient.sex}</p>
                                    </div>

                                    <div className="main-result">
                                        <h2>{result.results.sort((a,b)=>b.probability-a.probability)[0].description}</h2>
                                        <p className="prob">{(result.results.sort((a,b)=>b.probability-a.probability)[0].probability * 100).toFixed(1)}%</p>
                                    </div>

                                    <div className="details-list">
                                        {result.results.map((r, i) => (
                                            <div key={i} className="detail-row">
                                                <span>{r.description}</span>
                                                <span>{(r.probability * 100).toFixed(1)}%</span>
                                            </div>
                                        ))}
                                    </div>
                                    <button onClick={() => setStep("form")} style={{backgroundColor: '#666'}}>Nouveau Patient</button>
                                </div>

                                {/* Panneau Droite : Chat */}
                                <div className="right-panel">
                                    <ChatComponent 
                                        messages={chatMessages}
                                        input={chatInput}
                                        setInput={setChatInput}
                                        category={chatCategory}
                                        setCategory={setChatCategory}
                                        language={chatLanguage}
                                        setLanguage={setChatLanguage}
                                        onSend={handleChatSubmit}
                                    />
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}

export default App;