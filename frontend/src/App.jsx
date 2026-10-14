import React, { useState, useEffect } from 'react';
import { Search, Info, BarChart2, CheckCircle2, ChevronRight, Activity, ArrowRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

export default function App() {
  const [activeTab, setActiveTab] = useState('ask');
  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f3f4f6' }}>
      {/* Sidebar */}
      <div style={{ width: '250px', backgroundColor: '#1f2937', color: 'white', padding: '20px' }}>
        <h1 style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '2rem' }}>AdaptKG-RAG</h1>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <button 
            onClick={() => setActiveTab('ask')}
            style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px', backgroundColor: activeTab === 'ask' ? '#374151' : 'transparent', border: 'none', color: 'white', textAlign: 'left', cursor: 'pointer', borderRadius: '5px' }}>
            <Search size={18} /> Ask Question
          </button>
          <button 
            onClick={() => setActiveTab('overview')}
            style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px', backgroundColor: activeTab === 'overview' ? '#374151' : 'transparent', border: 'none', color: 'white', textAlign: 'left', cursor: 'pointer', borderRadius: '5px' }}>
            <Info size={18} /> System Overview
          </button>
          <button 
            onClick={() => setActiveTab('evaluation')}
            style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px', backgroundColor: activeTab === 'evaluation' ? '#374151' : 'transparent', border: 'none', color: 'white', textAlign: 'left', cursor: 'pointer', borderRadius: '5px' }}>
            <BarChart2 size={18} /> Evaluation
          </button>
        </nav>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
        {activeTab === 'ask' && <AskTab />}
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'evaluation' && <EvaluationTab />}
      </div>
    </div>
  );
}

function AskTab() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const ask = async (e) => {
    e.preventDefault();
    if (!question) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to fetch answer');
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '1.8rem', marginBottom: '1.5rem', color: '#111827' }}>Ask the Knowledge Graph</h2>
      
      <form onSubmit={ask} style={{ display: 'flex', gap: '10px', marginBottom: '2rem' }}>
        <input 
          type="text" 
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. who directed [Kismet]?"
          style={{ flex: 1, padding: '12px 16px', borderRadius: '8px', border: '1px solid #d1d5db', fontSize: '1rem' }}
        />
        <button 
          type="submit" 
          disabled={loading}
          style={{ padding: '0 24px', backgroundColor: '#2563eb', color: 'white', border: 'none', borderRadius: '8px', fontSize: '1rem', cursor: loading ? 'not-allowed' : 'pointer' }}>
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      {error && <div style={{ padding: '15px', backgroundColor: '#fee2e2', color: '#991b1b', borderRadius: '8px' }}>{error}</div>}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h3 style={{ fontSize: '1.2rem', color: '#374151', marginBottom: '1rem' }}>Answer</h3>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#10b981' }}>
              {result.answer && result.answer.length > 0 ? result.answer.join(', ') : 'No answer found'}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
            <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <h3 style={{ fontSize: '1.1rem', color: '#374151', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity size={18}/> Adaptive Logic
              </h3>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                <li><strong>Predicted Hop:</strong> {result.predicted_hop}</li>
                <li><strong>Confidence:</strong> {( (result.confidence || 0) * 100 ).toFixed(1)}%</li>
                <li><strong>Selected Depth:</strong> {result.selected_retrieval_depth}</li>
                <li><strong>Used Fallback:</strong> {result.used_fallback ? 'Yes' : 'No'}</li>
                <li style={{ marginTop: '8px', color: '#2563eb' }}><i>{result.adaptive_decision_reason}</i></li>
              </ul>
            </div>
            <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <h3 style={{ fontSize: '1.1rem', color: '#374151', marginBottom: '1rem' }}>Retrieval Stats</h3>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#4b5563', fontSize: '0.95rem', lineHeight: '1.6' }}>
                <li><strong>Entities Detected:</strong> {result.detected_entities && result.detected_entities.length > 0 ? result.detected_entities.map(e => typeof e === 'string' ? e : (e.name || e.text || 'Entity')).join(', ') : 'None'}</li>
                <li><strong>Candidate Triples:</strong> {result.candidate_triple_count}</li>
                <li><strong>Retrieval Time:</strong> {(result.retrieval_time_ms || 0).toFixed(1)} ms</li>
                <li><strong>Answer Source:</strong> {result.answer_source}</li>
              </ul>
            </div>
          </div>

          <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h3 style={{ fontSize: '1.1rem', color: '#374151', marginBottom: '1rem' }}>Evidence Path</h3>
            <div style={{ backgroundColor: '#f8fafc', padding: '15px', borderRadius: '6px', fontFamily: 'monospace', fontSize: '1rem', whiteSpace: 'pre-wrap', border: '1px solid #e2e8f0' }}>
              {result.evidence_paths && result.evidence_paths.length > 0 
                ? (typeof result.evidence_paths[0] === 'string' 
                    ? result.evidence_paths[0] 
                    : (result.evidence_paths[0].explanation || JSON.stringify(result.evidence_paths[0]))) 
                : 'No evidence path available'}
            </div>
          </div>
          
          <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h3 style={{ fontSize: '1.1rem', color: '#374151', marginBottom: '1rem' }}>Top Ranked Triples</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {result.ranked_triples && result.ranked_triples.length > 0 ? (
                result.ranked_triples.slice(0, 5).map((t, idx) => (
                  <div key={idx} style={{ padding: '8px', backgroundColor: '#f3f4f6', borderRadius: '4px', fontSize: '0.9rem', color: '#4b5563' }}>
                    <span style={{ fontWeight: 'bold' }}>[{(t.relevance_score || 0).toFixed(3)}]</span> {t.subject} → {t.relation} → {t.object}
                  </div>
                ))
              ) : (
                <div style={{ color: '#6b7280', fontSize: '0.9rem' }}>No triples retrieved</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function OverviewTab() {
  const [stats, setStats] = useState(null);
  
  useEffect(() => {
    fetch('/api/stats').then(r => r.json()).then(setStats).catch(console.error);
  }, []);

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', color: '#111827' }}>
      <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>AdaptKG-RAG Overview</h2>
      <p style={{ fontSize: '1.1rem', color: '#4b5563', lineHeight: '1.6', marginBottom: '2rem' }}>
        An Adaptive Knowledge Graph Retrieval Framework for Explainable Multi-Hop Question Answering.
      </p>

      <div style={{ backgroundColor: 'white', padding: '25px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Problem Statement & Solution</h3>
        <p style={{ color: '#4b5563', lineHeight: '1.6', marginBottom: '1rem' }}>
          Fixed-depth subgraph retrieval either under-retrieves for complex questions (missing answers) or over-retrieves for simple questions (causing massive latency and noise).
        </p>
        <p style={{ color: '#4b5563', lineHeight: '1.6' }}>
          <strong>AdaptKG-RAG</strong> introduces a lightweight hop classifier (TF-IDF + LogReg) that predicts the question complexity before retrieval. Using a confidence threshold, it dynamically adapts the BFS retrieval depth, ensuring precise and efficient evidence gathering from the Knowledge Graph.
        </p>
      </div>

      <div style={{ backgroundColor: 'white', padding: '25px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Pipeline Architecture</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', color: '#374151' }}>
          {['Entity Detection', 'Hop Prediction', 'Depth Selection', 'BFS Retrieval', 'Triple Ranking', 'Rule-Based Answer Engine'].map((step, idx, arr) => (
            <React.Fragment key={idx}>
              <div style={{ padding: '8px 12px', backgroundColor: '#e0e7ff', borderRadius: '20px', fontSize: '0.9rem', fontWeight: '500' }}>
                {step}
              </div>
              {idx < arr.length - 1 && <ArrowRight size={20} style={{ alignSelf: 'center', color: '#9ca3af' }} />}
            </React.Fragment>
          ))}
        </div>
      </div>

      {stats && (
        <div style={{ backgroundColor: 'white', padding: '25px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '1.5rem' }}>MetaQA Dataset Statistics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1.5rem' }}>
            <div>
              <div style={{ fontSize: '0.9rem', color: '#6b7280', textTransform: 'uppercase' }}>Triples</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#2563eb' }}>{stats.num_triples.toLocaleString()}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.9rem', color: '#6b7280', textTransform: 'uppercase' }}>Entities</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#2563eb' }}>{stats.num_entities.toLocaleString()}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.9rem', color: '#6b7280', textTransform: 'uppercase' }}>Relations</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#2563eb' }}>{stats.num_relations.toLocaleString()}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function EvaluationTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/results')
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { console.error(e); setLoading(false); });
  }, []);

  if (loading) return <div>Loading evaluation results...</div>;
  if (!data || !data.evaluation_summary) return <div style={{ color: '#4b5563', padding: '20px' }}>No evaluation results available yet.</div>;

  const chartData = (data.evaluation_summary || [])
    .filter(d => d && (d.Method || d.Hop))
    .map(d => {
      const m = String(d.Method || d.Hop || '');
      const name = m === 'proposed' ? 'AdaptKG-RAG' : m.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      return {
        name,
        Accuracy: (parseFloat(d['Exact Match'] || 0) * 100),
        Triples: parseFloat(d['Avg Triples'] || 0),
        Time: parseFloat(d['Avg Time (ms)'] || 0)
      };
    });

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', color: '#111827' }}>
      <h2 style={{ fontSize: '1.8rem', marginBottom: '2rem' }}>Evaluation Results</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        <div style={{ backgroundColor: 'white', padding: '25px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <h3 style={{ fontSize: '1.1rem', marginBottom: '1.5rem', textAlign: 'center' }}>Exact Match Accuracy (%)</h3>
          <div style={{ height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} />
                <RechartsTooltip formatter={(val) => val.toFixed(1) + '%'} />
                <Bar dataKey="Accuracy" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div style={{ backgroundColor: 'white', padding: '25px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
          <h3 style={{ fontSize: '1.1rem', marginBottom: '1.5rem', textAlign: 'center' }}>Retrieval Efficiency (Avg Triples)</h3>
          <div style={{ height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <RechartsTooltip formatter={(val) => val.toFixed(0)} />
                <Bar dataKey="Triples" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div style={{ backgroundColor: 'white', padding: '25px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Classifier Performance</h3>
        {data.classifier_report && data.classifier_report.accuracy !== undefined ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', textAlign: 'center' }}>
             <div style={{ padding: '15px', backgroundColor: '#f3f4f6', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.9rem', color: '#6b7280', textTransform: 'uppercase', marginBottom: '5px' }}>Accuracy</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#374151' }}>{(data.classifier_report.accuracy * 100).toFixed(2)}%</div>
             </div>
             <div style={{ padding: '15px', backgroundColor: '#f3f4f6', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.9rem', color: '#6b7280', textTransform: 'uppercase', marginBottom: '5px' }}>Precision</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#374151' }}>{(data.classifier_report['macro avg'].precision * 100).toFixed(2)}%</div>
             </div>
             <div style={{ padding: '15px', backgroundColor: '#f3f4f6', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.9rem', color: '#6b7280', textTransform: 'uppercase', marginBottom: '5px' }}>Recall</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#374151' }}>{(data.classifier_report['macro avg'].recall * 100).toFixed(2)}%</div>
             </div>
             <div style={{ padding: '15px', backgroundColor: '#f3f4f6', borderRadius: '6px' }}>
                <div style={{ fontSize: '0.9rem', color: '#6b7280', textTransform: 'uppercase', marginBottom: '5px' }}>F1 Score</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#374151' }}>{(data.classifier_report['macro avg']['f1-score'] * 100).toFixed(2)}%</div>
             </div>
          </div>
        ) : (
          <div style={{ color: '#6b7280' }}>Classifier metrics unavailable</div>
        )}
      </div>
    </div>
  );
}
