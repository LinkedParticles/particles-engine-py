/*
 * A minimal DOM for the golden capture (step 1): just what the
 * curation feed touches, so its output can be recorded in Node with no
 * browser and no new dependency. It is not a general DOM.
 */

class Text {
  constructor(text) {
    this.nodeType = 3;
    this.parentNode = null;
    this._text = String(text);
  }
  get textContent() {
    return this._text;
  }
  set textContent(v) {
    this._text = String(v);
  }
  remove() {
    if (this.parentNode) this.parentNode._removeChild(this);
  }
}

class ClassList {
  constructor(el) {
    this.el = el;
  }
  _list() {
    return this.el.className.split(/\s+/).filter(Boolean);
  }
  add(...names) {
    const list = this._list();
    for (const n of names) if (!list.includes(n)) list.push(n);
    this.el.className = list.join(" ");
  }
  remove(...names) {
    this.el.className = this._list().filter((n) => !names.includes(n)).join(" ");
  }
  contains(name) {
    return this._list().includes(name);
  }
}

export class Element {
  constructor(tag) {
    this.nodeType = 1;
    this.tagName = tag.toUpperCase();
    this.className = "";
    this.childNodes = [];
    this.parentNode = null;
    this.style = {};
    this.classList = new ClassList(this);
    this.listeners = {};
    this.attrs = {};
  }
  get children() {
    return this.childNodes.filter((n) => n.nodeType === 1);
  }
  get textContent() {
    return this.childNodes.map((n) => n.textContent).join("");
  }
  set textContent(v) {
    this.childNodes = [];
    if (v !== "") this.appendChild(new Text(v));
  }
  set innerHTML(v) {
    if (v !== "") throw new Error("the golden DOM only supports clearing innerHTML");
    this.childNodes = [];
  }
  _adopt(node) {
    if (typeof node === "string") node = new Text(node);
    if (node.parentNode) node.parentNode._removeChild(node);
    node.parentNode = this;
    return node;
  }
  _removeChild(node) {
    this.childNodes = this.childNodes.filter((n) => n !== node);
    node.parentNode = null;
  }
  appendChild(node) {
    this.childNodes.push(this._adopt(node));
    return node;
  }
  append(...nodes) {
    for (const n of nodes) this.appendChild(n);
  }
  prepend(...nodes) {
    this.childNodes.unshift(...nodes.map((n) => this._adopt(n)));
  }
  replaceChildren(...nodes) {
    this.childNodes = [];
    this.append(...nodes);
  }
  remove() {
    if (this.parentNode) this.parentNode._removeChild(this);
  }
  replaceWith(node) {
    const parent = this.parentNode;
    if (!parent) return;
    const adopted = parent._adopt(node);
    parent.childNodes.splice(parent.childNodes.indexOf(this), 1, adopted);
    this.parentNode = null;
  }
  addEventListener(type, fn) {
    (this.listeners[type] ??= []).push(fn);
  }
  dispatch(type, event) {
    for (const fn of this.listeners[type] ?? []) fn(event);
  }
  querySelector(selector) {
    if (!selector.startsWith(".")) throw new Error(`unsupported selector ${selector}`);
    const cls = selector.slice(1);
    for (const c of this.children) {
      if (c.classList.contains(cls)) return c;
      const hit = c.querySelector(selector);
      if (hit) return hit;
    }
    return null;
  }
  querySelectorAll(selector) {
    if (!selector.startsWith(".")) throw new Error(`unsupported selector ${selector}`);
    const cls = selector.slice(1);
    const out = [];
    const walk = (el) => {
      for (const c of el.children) {
        if (c.classList.contains(cls)) out.push(c);
        walk(c);
      }
    };
    walk(this);
    return out;
  }
}

/** The few attributes the feed sets, recorded under their property names. */
for (const prop of ["title", "disabled", "href", "target", "rel", "type", "checked", "selected", "value"]) {
  Object.defineProperty(Element.prototype, prop, {
    get() {
      return this.attrs[prop];
    },
    set(v) {
      this.attrs[prop] = v;
    },
  });
}

export const document = {
  createElement: (tag) => new Element(tag),
  createTextNode: (text) => new Text(text),
};

/**
 * A stable, plain-data picture of a subtree: an element is
 * `[tag, className, attrs, ...children]`, a text node is its string. Event
 * handlers are left out; `onclick` shows as the attribute `onclick: true`.
 */
export function dump(node) {
  if (node.nodeType === 3) return node.textContent;
  const attrs = { ...node.attrs };
  if (typeof node.onclick === "function") attrs.onclick = true;
  if (typeof node.onchange === "function") attrs.onchange = true;
  if (node.style && node.style.opacity) attrs.opacity = node.style.opacity;
  const out = [node.tagName.toLowerCase(), node.className, attrs];
  for (const c of node.childNodes) out.push(dump(c));
  return out;
}
