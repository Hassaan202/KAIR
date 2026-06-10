import { useState } from 'react'

/**
 * ArrayEditor — editable list of numbers
 */
export function ArrayEditor({ value, onChange, integer = true, label }) {
  const update = (i, v) => {
    const next = [...value]
    next[i] = integer ? parseInt(v) || 0 : parseFloat(v) || 0
    onChange(next)
  }
  const add = () => onChange([...value, integer ? 6 : 1.0])
  const remove = (i) => onChange(value.filter((_, idx) => idx !== i))

  return (
    <div className="form-group">
      {label && <label>{label}</label>}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
        {value.map((v, i) => (
          <input
            key={i}
            type="number"
            className="num-input"
            style={{ width: '60px' }}
            value={v}
            onChange={(e) => update(i, e.target.value)}
            step={integer ? 1 : 0.1}
          />
        ))}
        <button type="button" className="btn btn-ghost icon-btn" onClick={add}>+</button>
        {value.length > 1 && (
          <button type="button" className="btn btn-ghost icon-btn" onClick={() => remove(value.length - 1)}>−</button>
        )}
      </div>
    </div>
  )
}

/**
 * BoolToggle — labelled toggle switch
 */
export function BoolToggle({ label, value, onChange, hint }) {
  return (
    <div className="form-group">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <label className="switch">
          <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
          <span className="track" />
        </label>
        <span style={{ fontSize: 13, color: 'var(--ink-2)' }}>
          {label}
          {hint && <span className="hint">{hint}</span>}
        </span>
      </div>
    </div>
  )
}

/**
 * SelectField — labelled <select>
 */
export function SelectField({ label, value, onChange, options, hint }) {
  return (
    <div className="form-group">
      <label>{label}{hint && <span className="hint">{hint}</span>}</label>
      <select className="text-input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value ?? o} value={o.value ?? o}>{o.label ?? o}</option>
        ))}
      </select>
    </div>
  )
}

/**
 * TextField — labelled text input
 */
export function TextField({ label, value, onChange, placeholder, hint, mono }) {
  return (
    <div className="form-group">
      <label>{label}{hint && <span className="hint">{hint}</span>}</label>
      <input
        type="text"
        className="text-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={mono ? { fontFamily: 'var(--font-mono)', fontSize: 12 } : {}}
      />
    </div>
  )
}

/**
 * NumberField — labelled number input
 */
export function NumberField({ label, value, onChange, min, max, step = 1, hint }) {
  return (
    <div className="form-group">
      <label>{label}{hint && <span className="hint">{hint}</span>}</label>
      <input
        type="number"
        className="num-input"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
      />
    </div>
  )
}

/**
 * Collapsible section wrapper
 */
export function CollapsibleSection({ title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden', marginBottom: 16 }}>
      <button type="button" className="acc-head" onClick={() => setOpen((v) => !v)}>
        <span className="acc-title">{title}</span>
        <span style={{ color: 'var(--ink-3)', fontSize: 11 }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && <div className="acc-body">{children}</div>}
    </div>
  )
}
