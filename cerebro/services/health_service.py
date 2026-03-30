import httpx

class HealthService:
    @staticmethod
    async def check_qdrant(rag_service):
        try:
            await rag_service.get_collections()
            return "ok"
        except:
            return "error"

    @staticmethod
    async def check_langfuse(url):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{url}/api/public/health")
                return "ok" if response.status_code == 200 else "error"
        except:
            return "error"

    @staticmethod
    async def check_llm_models(models):
        results = {}

        async with httpx.AsyncClient(timeout=2.0) as client:
            for name, model in models.items():
                try:
                    r = await client.get(f"{model.llm.base_url}/models")
                    results[name] = "ok" if r.status_code == 200 else "error"
                except:
                    results[name] = "error"

        return results