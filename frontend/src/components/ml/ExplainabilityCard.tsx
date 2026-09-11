import { useFilterStore } from '../../store/useFilterStore';

export function ExplainabilityCard() {
  const predictionOpen = useFilterStore((state) => state.predictionOpen);
  const prediction = useFilterStore((state) => state.prediction);
  const factors = prediction?.explanation.top_contributing_factors ?? [];
  return <div className="explain-card"><div className="panel-heading"><div><p className="eyebrow">MODEL EXPLAINABILITY</p><h3>Why this location?</h3></div><span className="muted">COUNTERFACTUAL IMPACT</span></div>{!predictionOpen || !prediction ? <div className="empty-explain"><strong>Run a location analysis to see model-derived factors.</strong><span>These are local sensitivity estimates from the actual prediction, not SHAP values or static feature percentages.</span></div> : <>{prediction.training_data_coverage?.warning && <div className="coverage-warning">{prediction.training_data_coverage.warning}</div>}{factors.length === 0 ? <p className="explain-hint">No material feature contributions were returned for this location.</p> : <div className="factor-list">{factors.map((factor) => <div className="factor" key={factor.feature}><div className="factor-top"><strong>{factor.feature}</strong><b className={factor.impact === 'increased' ? 'positive' : 'negative'}>{factor.impact === 'increased' ? '+' : '-'}{Math.abs(factor.contribution_tonnes).toLocaleString('en-IN')} t</b></div><span>{factor.value}</span><small>{factor.reason}</small></div>)}</div>}</>}</div>;
}
