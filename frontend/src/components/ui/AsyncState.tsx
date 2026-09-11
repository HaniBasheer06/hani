export function LoadingBlock() { return <div className="loading-block"><span /><span /><span /></div>; }
export function ErrorState({ message }: { message: string }) { return <div className="error-state">{message} <button onClick={() => window.location.reload()}>Retry</button></div>; }
